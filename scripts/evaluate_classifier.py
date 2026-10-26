"""Evaluate the species classifier on the cached crops of a split (labelled boxes, no
detector). Settings: configs/classifier.yaml (`evaluate`, plus `data`, `crops`, `run_name` and
`output_dir` to find the run and the crops).

Writes to <run>/eval/<split>_<checkpoint>/: metrics.json (accuracy and balanced accuracy),
per_class.csv, confusion_matrix.csv/.png, taxonomic_levels.csv (share right at species,
genus, family, order and, with `evaluate.rules`, role level), accuracy_by_size.csv,
accuracy_by_dataset.csv (for splits that combine several datasets) and predictions.csv
(one row per crop). Compare taxonomic_levels.csv with the Mask R-CNN's: both
count the plants that were found, here all of them.

Plants with a partial label (e.g. PhenoBench's weeds: not sugar beet) have no class to be
right about, so they are left out of all of these. partial_labels.csv gives instead, per
dataset and label, the share whose predicted class is one the label allows (`label_fit`)
and the classes most often predicted; partial_predictions.csv has one row per such crop.
"""

import csv
import json
from collections import Counter, defaultdict
from itertools import pairwise

import numpy as np
from torch.utils.data import DataLoader

from weedseg.classifier.crops import cache_dir, read_index
from weedseg.classifier.data import CropDataset
from weedseg.classifier.engine import accuracy_metrics, label_fit, predict_probs
from weedseg.classifier.model import load_classifier
from weedseg.config import load_config
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.data.combine import describe_classes
from weedseg.decisions import level_accuracy, load_rules, role_of
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.engine import select_device
from weedseg.maskrcnn.run import checkpoint_path, eval_dir, run_name
from weedseg.paths import resolve
from weedseg.visualize import plot_confusion_matrix

CONFIG = "configs/classifier.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    settings = config["evaluate"]
    device = select_device(config["device"])
    amp = config["train"]["amp"] and device.type == "cuda"
    checkpoint_file = checkpoint_path(config)
    model, checkpoint = load_classifier(checkpoint_file, device)
    classes, crops = checkpoint["classes"], checkpoint["config"]["crops"]
    print(f"{checkpoint_file} (epoch {checkpoint['epoch']}), device {device}")

    split = settings["split"]
    folder = cache_dir(resolve(config["data"]["dir"]), crops)
    index = folder / f"index_{split}.csv"
    if not index.exists():
        raise SystemExit(f"No crop cache {index}; run scripts/prepare_crops.py first")
    rows = read_index(index)
    dataset = CropDataset(rows, folder / split, classes, crops)
    loader = DataLoader(dataset, config["train"]["eval_batch_size"],
                        num_workers=config["data"]["num_workers"])
    probs, labels = predict_probs(model, loader, device, amp)
    partly = labels < 0  # partial labels: scored apart, in partial_labels.csv
    partial_rows = [row for row, p in zip(rows, partly) if p]
    partial_probs = probs[partly]
    partial_allowed = [allowed for allowed in dataset.allowed if allowed]
    rows = [row for row, p in zip(rows, partly) if not p]
    probs, labels = probs[~partly], labels[~partly]
    predicted = probs.argmax(1)

    out_dir = eval_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = accuracy_metrics(probs, labels, len(classes)) | {"crops": len(labels)}
    if partial_rows:
        metrics |= {"partial_crops": len(partial_rows),
                    "partial_label_fit": label_fit(partial_probs.argmax(1), partial_allowed)}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    k = len(classes)
    matrix = np.zeros((k + 1, k + 1), np.int64)  # last row/column: background, unused here
    np.add.at(matrix, (labels, predicted), 1)
    names = [*classes, "background"]
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth \\ predicted", *classes])
        writer.writerows([name, *row] for name, row in zip(classes, matrix[:k, :k].tolist()))
    plot_confusion_matrix(matrix[:k, :k], classes, out_dir / "confusion_matrix.png",
                          f"{run_name(config)}, {split} crops")

    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "crops", "accuracy"])
        for i, name in enumerate(classes):
            total = int(matrix[i, :k].sum())
            writer.writerow([name, total, matrix[i, i] / total if total else None])

    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    if all(name == OTHER_CATEGORY or name in taxa for name in classes):
        roles = None
        if settings["rules"]:
            rules = load_rules(resolve(settings["rules"]), taxa)
            roles = {name: role_of(name, rules, taxa) for name in classes}
        levels = level_accuracy(matrix, names, taxa, roles)
        other_pairs = levels.pop("other_plant_pairs")
        with open(out_dir / "taxonomic_levels.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["level", "plants", "correct", "accuracy"])
            writer.writerows([level, v["plants"], v["correct"], v["accuracy"]]
                             for level, v in levels.items())
        print("taxonomic levels: " + "  ".join(
            f"{level} {v['accuracy']:.1%}" for level, v in levels.items()
            if v["accuracy"] is not None) + f"  ({other_pairs} pairs with {OTHER_CATEGORY} "
                                            "left out)")

    edges = settings["size_bins_mm"]
    sizes = np.array([row["size_mm"] if row["size_mm"] is not None else np.nan for row in rows])
    with open(out_dir / "accuracy_by_size.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["size_mm", "crops", "accuracy"])
        bounds = [(None, edges[0]), *pairwise(edges), (edges[-1], None)]
        for low, high in bounds:
            inside = ~np.isnan(sizes)
            if low is not None:
                inside &= sizes >= low
            if high is not None:
                inside &= sizes < high
            label = f"<{high}" if low is None else f">={low}" if high is None else f"{low}-{high}"
            total = int(inside.sum())
            accuracy = float((predicted[inside] == labels[inside]).mean()) if total else None
            writer.writerow([label, total, accuracy])
            if total:
                print(f"size {label} mm: {total} crops, accuracy {accuracy:.1%}")

    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) > 1:
        with open(out_dir / "accuracy_by_dataset.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "crops", "accuracy", "balanced_accuracy"])
            for name in datasets:
                inside = np.array([row["dataset"] == name for row in rows])
                values = accuracy_metrics(probs[inside], labels[inside], len(classes))
                writer.writerow([name or "(none)", int(inside.sum()), values["accuracy"],
                                 values["balanced_accuracy"]])
                print(f"dataset {name or '(none)'}: {int(inside.sum())} crops, accuracy "
                      f"{values['accuracy']:.1%}, balanced accuracy "
                      f"{values['balanced_accuracy']:.1%}")

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "image_id", "true", "predicted", "probability",
                         "size_mm", "dataset"])
        for row, label, guess, p in zip(rows, labels, predicted, probs):
            writer.writerow([row["annotation_id"], row["image_id"], classes[label],
                             classes[guess], round(float(p[guess]), 4), row["size_mm"],
                             row["dataset"]])
    if partial_rows:
        write_partial(out_dir, partial_rows, partial_probs, classes)
    print(f"accuracy {metrics['accuracy']:.1%}, balanced accuracy "
          f"{metrics['balanced_accuracy']:.1%} on {len(labels)} crops -> {out_dir}")


def write_partial(out_dir, rows: list[dict], probs: np.ndarray, classes: list[str]) -> None:
    """partial_labels.csv: per dataset and partial label, the crops, the share whose
    predicted class the label allows, and the classes most often predicted.
    partial_predictions.csv: one row per crop."""
    predicted = [classes[k] for k in probs.argmax(1)]
    groups = defaultdict(list)  # (dataset, EPPO code, allowed classes) -> crop positions
    for i, row in enumerate(rows):
        groups[row["dataset"], row["eppo"], tuple(row["allowed"])].append(i)
    with open(out_dir / "partial_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "eppo", "allowed", "crops", "label_fit", "most_predicted"])
        for (name, eppo, allowed), members in sorted(groups.items(),
                                                     key=lambda kv: (kv[0][0], -len(kv[1]))):
            fit = sum(predicted[i] in allowed for i in members) / len(members)
            top = Counter(predicted[i] for i in members).most_common(3)
            most = "; ".join(f"{label} {n / len(members):.0%}" for label, n in top)
            allowed_text = describe_classes(list(allowed), classes)
            writer.writerow([name, eppo, allowed_text, len(members), fit, most])
            print(f"partial label {name or '(none)'} {eppo} ({allowed_text[:60]}): "
                  f"{len(members)} crops, {fit:.1%} fit the label; most predicted {most}")
    with open(out_dir / "partial_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "image_id", "eppo", "allowed", "predicted",
                         "probability", "fits_label", "size_mm", "dataset"])
        for row, guess, p in zip(rows, predicted, probs):
            writer.writerow([row["annotation_id"], row["image_id"], row["eppo"],
                             describe_classes(row["allowed"], classes), guess,
                             round(float(p.max()), 4), guess in row["allowed"],
                             row["size_mm"], row["dataset"]])


if __name__ == "__main__":
    main()
