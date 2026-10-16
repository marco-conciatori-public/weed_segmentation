"""Train the Mask R-CNN baseline. Settings: configs/maskrcnn.yaml (all sections but `evaluate`).

Outputs go to <output_dir>/<run_name>/: last.pt (to resume), best.pt (best val segm AP),
metrics.jsonl (one line per epoch) and tensorboard/ (`uv run tensorboard --logdir runs`).
Re-running continues from last.pt; to start over, change run_name.
"""

import json

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from weedseg.config import load_config
from weedseg.data.coco import ignore_small_plants
from weedseg.evaluation import evaluate_coco
from weedseg.maskrcnn.data import (
    CocoInstanceDataset,
    build_augmentation,
    collate,
    load_coco,
    model_categories,
)
from weedseg.maskrcnn.engine import (
    predict,
    save_atomic,
    seed_everything,
    select_device,
    train_one_epoch,
    warmup_cosine,
)
from weedseg.maskrcnn.model import build_model
from weedseg.paths import resolve

CONFIG = "configs/maskrcnn.yaml"


def main() -> None:
    config = load_config(CONFIG)
    data, train, model_config = config["data"], config["train"], config["model"]
    run_dir = resolve(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    device = select_device(config["device"])
    amp = train["amp"] and device.type == "cuda"
    print(f"Run {run_dir}, device {device}{' (mixed precision)' if amp else ''}")

    train_coco, val_coco = (ignore_small_plants(load_coco(resolve(data[split])),
                                                data["min_plant_size_mm"])
                            for split in ("train", "val"))
    categories = model_categories(train_coco)
    if [c["name"] for c in model_categories(val_coco)] != [c["name"] for c in categories]:
        raise SystemExit("Train and val files have different categories")
    train_set = CocoInstanceDataset(train_coco, build_augmentation(config["augment"]),
                                    max_images=data["max_train_images"], seed=config["seed"])
    val_set = CocoInstanceDataset(val_coco, with_targets=False,
                                  max_images=data["max_val_images"], seed=config["seed"])
    loader_options = {"num_workers": data["num_workers"], "collate_fn": collate,
                      "pin_memory": device.type == "cuda",
                      "persistent_workers": data["num_workers"] > 0}
    train_loader = DataLoader(train_set, train["batch_size"], shuffle=True, drop_last=True,
                              **loader_options)
    val_loader = DataLoader(val_set, train["eval_batch_size"], **loader_options)
    print(f"{len(train_set)} train / {len(val_set)} val images, {len(categories)} categories: "
          f"{', '.join(c['name'] for c in categories)}")

    model = build_model(len(categories) + 1, model_config).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=train["lr"], momentum=train["momentum"],
                                weight_decay=train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(train["epochs"] * len(train_loader), train["warmup_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, step, best_ap = 0, 0, -1.0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if train["resume"] and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config"]["model"] != model_config:
            raise SystemExit(f"The model settings differ from those of {last_path}; "
                             "change run_name to train a new model")
        if checkpoint["config"]["data"].get("min_plant_size_mm") != data["min_plant_size_mm"]:
            raise SystemExit(f"min_plant_size_mm differs from that of {last_path}; "
                             "change run_name to train a new model")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step, best_ap = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best_ap"]
        print(f"Resuming after epoch {checkpoint['epoch']} (best val segm AP {best_ap:.3f})")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step or None)

    def log_step(values: dict[str, float]) -> None:
        nonlocal step
        step += 1
        if step % train["log_every"] == 0:
            for name, value in values.items():
                writer.add_scalar(f"train/{name}", value, step)

    for epoch in range(start_epoch, train["epochs"]):
        losses = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, amp,
                                 train["grad_clip"], epoch, log_step)
        record = {"epoch": epoch, "step": step} | {f"train_{k}": v for k, v in losses.items()}

        if (epoch + 1) % train["eval_every"] == 0 or epoch == train["epochs"] - 1:
            results = predict(model, val_loader, categories, device, amp,
                              model_config["mask_threshold"])
            metrics = evaluate_coco(val_coco, results, [i["id"] for i in val_set.images],
                                    max_detections=model_config["detections_per_image"])
            for iou_type, values in metrics.items():
                for name in ("AP", "AP50"):
                    record[f"val_{iou_type}_{name}"] = values["summary"][name]
                    writer.add_scalar(f"val/{iou_type}_{name}", values["summary"][name], step)
            if metrics["segm"]["summary"]["AP"] > best_ap:
                best_ap = metrics["segm"]["summary"]["AP"]
                save_atomic({"model": model.state_dict(), "categories": categories,
                             "config": config, "epoch": epoch, "metrics": metrics}, best_path)
                record["best"] = True

        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "categories": categories, "config": config, "epoch": epoch,
                     "step": step, "best_ap": best_ap}, last_path)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        writer.flush()
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in record.items()))
    writer.close()
    print(f"Done. Best val segm AP {best_ap:.3f} -> {best_path}")


if __name__ == "__main__":
    main()
