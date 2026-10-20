"""Evaluation outputs of instance predictions (COCO results) on a species or plant split,
shared by scripts/evaluate_maskrcnn.py and scripts/predict_twostage.py so both pipelines are
scored by the same code.

`write_evaluation` writes to one folder: predictions.json, metrics.json (COCO bbox and segm),
per_class.csv, confusion_matrix.csv/.png, taxonomic_levels.csv and, with `rules`,
decisions.csv/.json (both only for species-level classes), previews/ (ground truth left,
predictions right) and evaluation.json (what was evaluated, read by scripts/review_errors.py).
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weedseg.data.coco import IGNORE_CATEGORY, OTHER_CATEGORY
from weedseg.decisions import (
    ACTION_COLUMNS,
    TRUE_ROWS,
    apply_decisions,
    decision_matrix,
    decision_summary,
    level_accuracy,
    load_rules,
    role_of,
)
from weedseg.eppo import load_taxa
from weedseg.evaluation import confusion_matrix, evaluate_coco
from weedseg.paths import project_root, resolve
from weedseg.visualize import draw_annotations, plot_confusion_matrix, put_label


def write_evaluation(coco: dict[str, Any], results: list[dict[str, Any]],
                     settings: dict[str, Any], out_dir: Path, title: str, max_detections: int,
                     image_dir: Path, split_file: Path, seed: int) -> dict[str, Any]:
    """Write every evaluation output of `results` on `coco` (already restricted to the
    evaluated images) to `out_dir`, and return the COCO metrics.

    `settings` is an `evaluate` config section: score_threshold, iou_threshold, match_by,
    rules, eppo_taxa, num_previews, preview_score_threshold, split and max_images.
    `split_file` and `seed` are recorded so the error review finds the same images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"split_file": _relative(split_file), "seed": seed, "settings": settings},
                  f, indent=2)
    metrics = evaluate_coco(coco, results, max_detections=max_detections, verbose=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_per_class(metrics, out_dir / "per_class.csv")

    matrix, names = confusion_matrix(coco, results, score_threshold=settings["score_threshold"],
                                     iou_threshold=settings["iou_threshold"],
                                     match_by=settings["match_by"])
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth \\ predicted", *names])
        writer.writerows([name, *row] for name, row in zip(names, matrix.tolist()))
    plot_confusion_matrix(matrix, names, out_dir / "confusion_matrix.png",
                          f"{title}, {settings['split']} (score >= {settings['score_threshold']}, "
                          f"{settings['match_by']} IoU >= {settings['iou_threshold']})")

    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    if all(name == OTHER_CATEGORY or name in taxa for name in names[:-1]):
        write_levels(matrix, names, settings, out_dir / "taxonomic_levels.csv")
        if settings["rules"]:
            write_decisions(coco, results, settings, out_dir)
    else:
        print("taxonomic levels and decisions: skipped, the classes are not EPPO codes")

    if settings["num_previews"]:
        save_previews(coco, results, image_dir, out_dir / "previews", settings["num_previews"],
                      settings["preview_score_threshold"], seed)
    for iou_type, values in metrics.items():
        summary = values["summary"]
        print(f"{iou_type}: AP {summary['AP']:.3f}  AP50 {summary['AP50']:.3f}")
    print(f"-> {out_dir}")
    return metrics


def _relative(path: Path) -> str:
    """`path` relative to the project root when inside it (portable between machines)."""
    try:
        return Path(path).resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def write_decisions(coco: dict, results: list[dict], settings: dict, out_dir: Path) -> None:
    rules_path = resolve(settings["rules"])
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    rules = load_rules(rules_path, taxa)
    roles = {c["name"]: role_of(c["name"], rules, taxa) for c in coco["categories"]
             if c["name"] != IGNORE_CATEGORY}
    matrix, names = confusion_matrix(
        coco, apply_decisions(results, coco["categories"], roles, rules.defer_below),
        score_threshold=settings["score_threshold"], iou_threshold=settings["iou_threshold"],
        match_by=settings["match_by"])
    grouped = decision_matrix(matrix, names, roles)
    with open(out_dir / "decisions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true role \\ action", *ACTION_COLUMNS])
        writer.writerows([role, *row] for role, row in zip(TRUE_ROWS, grouped.tolist()))
    summary = decision_summary(grouped)
    with open(out_dir / "decisions.json", "w", encoding="utf-8") as f:
        json.dump({"rules": str(rules_path), "summary": summary, "roles": roles}, f, indent=2)
    print(f"decisions ({rules_path.name}): " + "  ".join(
        f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()))


def write_levels(matrix: np.ndarray, names: list[str], settings: dict, path: Path) -> None:
    """Accuracy per taxonomic level of the plants found, from the confusion matrix (without
    the rules' defer_below: it describes the model). Needs species-level classes; the role
    level needs `rules`."""
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    roles = None
    if settings["rules"]:
        rules = load_rules(resolve(settings["rules"]), taxa)
        roles = {name: role_of(name, rules, taxa) for name in names[:-1]}
    levels = level_accuracy(matrix, names, taxa, roles)
    other_pairs = levels.pop("other_plant_pairs")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "plants", "correct", "accuracy"])
        writer.writerows([level, v["plants"], v["correct"], v["accuracy"]]
                         for level, v in levels.items())
    print("taxonomic levels (plants found, class right at): " + "  ".join(
        f"{level} {v['accuracy']:.1%}" for level, v in levels.items()
        if v["accuracy"] is not None)
          + f"  ({other_pairs} pairs with {OTHER_CATEGORY} left out)")


def write_per_class(metrics: dict, path: Path) -> None:
    rows = defaultdict(dict)
    for iou_type, values in metrics.items():
        for entry in values["per_class"]:
            row = rows[entry["category"]]
            row["category"], row["gt_instances"] = entry["category"], entry["gt_instances"]
            row[f"{iou_type}_AP"], row[f"{iou_type}_AP50"] = entry["AP"], entry["AP50"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["category", "gt_instances", *(
            f"{t}_{m}" for t in metrics for m in ("AP", "AP50"))])
        writer.writeheader()
        writer.writerows(rows.values())


def save_previews(coco: dict, results: list[dict], image_dir: Path, out_dir: Path,
                  count: int, score_threshold: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gts, dts = defaultdict(list), defaultdict(list)
    for annotation in coco["annotations"]:
        gts[annotation["image_id"]].append(annotation)
    for result in results:
        if result["score"] >= score_threshold:
            dts[result["image_id"]].append(result)
    images = [i for i in coco["images"] if gts[i["id"]]]
    for info in random.Random(seed).sample(images, min(count, len(images))):
        image = cv2.imread(str(image_dir / info["file_name"]))
        left = draw_annotations(image, gts[info["id"]], coco["categories"], draw_stems=False)
        right = draw_annotations(image, dts[info["id"]], coco["categories"], draw_stems=False)
        put_label(left, f"ground truth  {info['file_name']}", (0, 0), (0, 0, 0), 0.8, 2)
        put_label(right, f"prediction (score >= {score_threshold})", (0, 0), (0, 0, 0), 0.8, 2)
        side_by_side = np.hstack([left, np.full((image.shape[0], 8, 3), 255, np.uint8), right])
        cv2.imwrite(str(out_dir / f"{Path(info['file_name']).stem}.jpg"), side_by_side,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

