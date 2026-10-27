"""Train the species classifier on cached plant crops. Settings: configs/classifier.yaml
(all sections but `evaluate`). Run scripts/prepare_crops.py first.

With `data.partial_labels`, plants whose label only rules out some classes (e.g.
PhenoBench's weeds: not sugar beet) train as "one of the allowed classes" (PartialLabelLoss)
and are scored on val by the share whose predicted class fits (`val_partial_label_fit`).

Outputs go to <output_dir>/<run_name>/: last.pt (the final epoch, also to resume), best.pt
(best val accuracy), metrics.jsonl (one line per epoch) and tensorboard/ (`uv run tensorboard
--logdir runs`). The evaluation uses last.pt by default; at the end, a warning shows when the
last epoch is more than `train.warn_last_below_best` below the best.
Re-running continues from last.pt, which is also saved every `train.checkpoint_every` steps
(so a crash loses minutes, not an epoch); to start over, change run_name.
"""

import json
import random
from functools import partial as bind

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from weedseg.classifier.crops import cache_dir, read_index
from weedseg.classifier.data import CropDataset, sample_weights
from weedseg.classifier.engine import (
    PartialLabelLoss,
    accuracy_metrics,
    label_fit,
    last_below_best,
    predict_probs,
    train_one_epoch,
)
from weedseg.classifier.model import build_classifier, parameter_groups
from weedseg.config import load_config
from weedseg.maskrcnn.data import load_coco, model_categories
from weedseg.maskrcnn.engine import (
    EpochSampler,
    save_atomic,
    seed_everything,
    select_device,
    warmup_cosine,
)
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
        if not data["partial_labels"]:
            rows[split] = [row for row in rows[split] if not row["allowed"]]
        if limit is not None and limit < len(rows[split]):
            rows[split] = random.Random(config["seed"]).sample(rows[split], limit)
    train_set = CropDataset(rows["train"], folder / "train", classes, crops, config["augment"])
    val_set = CropDataset(rows["val"], folder / "val", classes, crops)
    weights = sample_weights(train_set.balance_keys(), train["balance"])
    sampler = EpochSampler(len(train_set), config["seed"], weights)
    loader_options = {"num_workers": data["num_workers"], "pin_memory": device.type == "cuda",
                      "persistent_workers": data["num_workers"] > 0}
    train_loader = DataLoader(train_set, train["batch_size"], sampler=sampler, drop_last=True,
                              **loader_options)
    steps_per_epoch = len(train_set) // train["batch_size"]
    val_loader = DataLoader(val_set, train["eval_batch_size"], **loader_options)
    partial_counts = {name: sum(label < 0 for label in dataset.labels)
               for name, dataset in (("train", train_set), ("val", val_set))}
    print(f"{len(train_set)} train / {len(val_set)} val crops (with partial labels: "
          f"{partial_counts['train']} / {partial_counts['val']}), {len(classes)} classes: "
          f"{', '.join(classes)}")

    model = build_classifier(len(classes), config["model"], crops["input_size"]).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, train["lr_backbone"], train["lr_head"],
                                                   train["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(train["epochs"] * steps_per_epoch, train["warmup_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = PartialLabelLoss(train["label_smoothing"])

    start_epoch, step, best, partial = 0, 0, -1.0, None
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
        partial = checkpoint.get("partial")
        where = (f"at batch {partial['batches']} of epoch {start_epoch}" if partial
                 else f"after epoch {checkpoint['epoch']}")
        print(f"Resuming {where} (best val accuracy {best:.3f})")
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    writer = SummaryWriter(str(run_dir / "tensorboard"), purge_step=step or None)

    def log_step(values: dict[str, float]) -> None:
        nonlocal step
        step += 1
        if step % train["log_every"] == 0:
            for name, value in values.items():
                writer.add_scalar(f"train/{name}", value, step)

    def save_last(finished_epoch: int, partial: dict | None = None) -> None:
        """last.pt after `finished_epoch`, plus the state of the next epoch if under way."""
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                     "classes": classes, "config": config, "epoch": finished_epoch,
                     "step": step, "best": best, "partial": partial}, last_path)
        writer.flush()

    for epoch in range(start_epoch, train["epochs"]):
        sampler.set_epoch(epoch, partial["batches"] * train["batch_size"] if partial else 0)
        result = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion,
                                 device, amp, train["grad_clip"], epoch, log_step, partial,
                                 train["checkpoint_every"], bind(save_last, epoch - 1))
        partial = None
        record = {"epoch": epoch, "step": step} | {f"train_{k}": v for k, v in result.items()}
        probs, labels = predict_probs(model, val_loader, device, amp)
        metrics = accuracy_metrics(probs, labels, len(classes))
        if partial_counts["val"]:
            metrics["partial_label_fit"] = label_fit(
                probs[labels < 0].argmax(1), [a for a in val_set.allowed if a])
        for name, value in metrics.items():
            record[f"val_{name}"] = value
            writer.add_scalar(f"val/{name}", value, step)
        if record["val_accuracy"] > best:
            best = record["val_accuracy"]
            save_atomic({"model": model.state_dict(), "classes": classes, "config": config,
                         "epoch": epoch, "val_accuracy": best}, best_path)
            record["best"] = True
        save_last(epoch)
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        writer.flush()
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in record.items()))
    writer.close()
    print(f"Done. Best val accuracy {best:.3f} -> {best_path}")
    metrics_path = run_dir / "metrics.jsonl"
    records = ([json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line] if metrics_path.exists() else [])
    warning = last_below_best(records, train["warn_last_below_best"])
    if warning:
        print(warning)


if __name__ == "__main__":
    main()
