"""Train the species classifier on cached plant crops. Settings: configs/classifier.yaml
(all sections but `evaluate`). Run scripts/prepare_crops.py first.

Outputs go to <output_dir>/<run_name>/: last.pt (to resume), best.pt (best val accuracy),
metrics.jsonl (one line per epoch) and tensorboard/ (`uv run tensorboard --logdir runs`).
Re-running continues from last.pt; to start over, change run_name.
"""

import json
import random

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from weedseg.classifier.crops import cache_dir, read_index
from weedseg.classifier.data import CropDataset, sample_weights
from weedseg.classifier.engine import accuracy_metrics, predict_probs, train_one_epoch
from weedseg.classifier.model import build_classifier, parameter_groups
from weedseg.config import load_config
from weedseg.maskrcnn.data import load_coco, model_categories
from weedseg.maskrcnn.engine import save_atomic, seed_everything, select_device, warmup_cosine
from weedseg.maskrcnn.run import resolve_run_dir, split_file
from weedseg.paths import resolve

CONFIG = "configs/classifier.yaml"
# Settings a resumed run must share with the run it continues.
FIXED_SECTIONS = ("data", "crops", "model")


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    data, crops, train = config["data"], config["crops"], config["train"]
    run_dir = resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    device = select_device(config["device"])
    amp = train["amp"] and device.type == "cuda"
    print(f"Run {run_dir}, device {device}{' (mixed precision)' if amp else ''}")

    classes = [c["name"] for c in model_categories(load_coco(split_file(config, "train")))]
    folder = cache_dir(resolve(data["dir"]), crops)
    rows = {}
    for split, limit in (("train", data["max_train_crops"]), ("val", data["max_val_crops"])):
        index = folder / f"index_{split}.csv"
        if not index.exists():
            raise SystemExit(f"No crop cache {index}; run scripts/prepare_crops.py first")
        rows[split] = read_index(index)
        if limit is not None and limit < len(rows[split]):
            rows[split] = random.Random(config["seed"]).sample(rows[split], limit)
    train_set = CropDataset(rows["train"], folder / "train", classes, crops, config["augment"])
    val_set = CropDataset(rows["val"], folder / "val", classes, crops)
    weights = sample_weights(train_set.labels, train["balance"])
    sampler = WeightedRandomSampler(weights, len(weights)) if weights is not None else None
    loader_options = {"num_workers": data["num_workers"], "pin_memory": device.type == "cuda",
                      "persistent_workers": data["num_workers"] > 0}
    train_loader = DataLoader(train_set, train["batch_size"], shuffle=sampler is None,
                              sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(val_set, train["eval_batch_size"], **loader_options)
    print(f"{len(train_set)} train / {len(val_set)} val crops, {len(classes)} classes: "
          f"{', '.join(classes)}")

    model = build_classifier(len(classes), config["model"], crops["input_size"]).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, train["lr_backbone"], train["lr_head"],
                                                   train["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(train["epochs"] * len(train_loader), train["warmup_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train["label_smoothing"])

    start_epoch, step, best = 0, 0, -1.0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if train["resume"] and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        changed = [s for s in FIXED_SECTIONS if checkpoint["config"][s] != config[s]]
        if changed:
            raise SystemExit(f"The {', '.join(changed)} settings differ from those of "
                             f"{last_path}; change run_name to train a new model")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step, best = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best"]
        print(f"Resuming after epoch {checkpoint['epoch']} (best val accuracy {best:.3f})")
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
        result = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion,
                                 device, amp, train["grad_clip"], epoch, log_step)
        record = {"epoch": epoch, "step": step} | {f"train_{k}": v for k, v in result.items()}
        probs, labels = predict_probs(model, val_loader, device, amp)
        for name, value in accuracy_metrics(probs, labels, len(classes)).items():
            record[f"val_{name}"] = value
            writer.add_scalar(f"val/{name}", value, step)
        if record["val_accuracy"] > best:
            best = record["val_accuracy"]
            save_atomic({"model": model.state_dict(), "classes": classes, "config": config,
                         "epoch": epoch, "val_accuracy": best}, best_path)
            record["best"] = True
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "classes": classes, "config": config, "epoch": epoch, "step": step,
                     "best": best}, last_path)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        writer.flush()
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in record.items()))
    writer.close()
    print(f"Done. Best val accuracy {best:.3f} -> {best_path}")


if __name__ == "__main__":
    main()
