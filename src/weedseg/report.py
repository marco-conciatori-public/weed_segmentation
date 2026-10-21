"""Evaluation outputs of instance predictions (COCO results) on a species or plant split,
shared by scripts/evaluate_maskrcnn.py and scripts/predict_twostage.py so both pipelines are
scored by the same code.

`write_evaluation` writes to one folder: predictions.json, metrics.json (COCO bbox and segm),
per_class.csv, confusion_matrix.csv/.png, taxonomic_levels.csv and, with `rules`,
decisions.csv/.json (both only for species-level classes), by_dataset.csv (the main numbers
per source dataset, for splits that combine several), previews/ (ground truth left,
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
    species = all(name == OTHER_CATEGORY or name in taxa for name in names[:-1])
    if species:
        write_levels(matrix, names, settings, out_dir / "taxonomic_levels.csv")
        if settings["rules"]:
            write_decisions(coco, results, settings, out_dir)
    else:
        print("taxonomic levels and decisions: skipped, the classes are not EPPO codes")
    write_by_dataset(coco, results, settings, metrics, max_detections, species,
                     out_dir / "by_dataset.csv")

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
    _, rules, roles = _rules_and_roles(
        settings, [c["name"] for c in coco["categories"] if c["name"] != IGNORE_CATEGORY])
    matrix, names = confusion_matrix(
        coco, apply_decisions(results, coco["categories"], roles, rules.defer_below,
                              rules.plant_defer_below),
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
    taxa, _, roles = _rules_and_roles(settings, names[:-1])
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


def write_by_dataset(coco: dict, results: list[dict], settings: dict, metrics: dict,
                     max_detections: int, species: bool, path: Path) -> None:
    """One row per source dataset (the images' `dataset` field), and one for all of them:
    COCO AP, the plants found at the score threshold and, with species-level classes, the
    share of found plants whose species and role are right and the decision summary. Shows
    whether a dataset helps on the others, which the overall numbers can hide (a dense
    dataset weighs most in them). Nothing is written for a split from a single dataset.

    `metrics` are those of all images (evaluate_coco), so they aren't computed twice.
    """
    image_ids = defaultdict(list)
    for image in coco["images"]:
        image_ids[image.get("dataset") or "(none)"].append(image["id"])
    if len(image_ids) < 2:
        return
    taxa = roles = decided = None
    if species:
        names = [c["name"] for c in coco["categories"] if c["name"] != IGNORE_CATEGORY]
        taxa, rules, roles = _rules_and_roles(settings, names)
        if rules is not None:
            decided = apply_decisions(results, coco["categories"], roles, rules.defer_below,
                                      rules.plant_defer_below)

    def row(name: str, ids: list[int] | None, metrics: dict | None = None) -> dict[str, Any]:
        if metrics is None:
            metrics = evaluate_coco(coco, results, ids, max_detections=max_detections)
        thresholds = (settings["score_threshold"], settings["iou_threshold"],
                      settings["match_by"])
        matrix, names = confusion_matrix(coco, results, ids, *thresholds)
        plants, missed = int(matrix[:-1].sum()), int(matrix[:-1, -1].sum())
        values = {"dataset": name, "images": len(ids if ids is not None else coco["images"]),
                  "plants": plants}
        for iou_type, entry in metrics.items():
            values |= {f"{iou_type}_{key}": entry["summary"][key] for key in ("AP", "AP50")}
        values |= {"found": (plants - missed) / plants if plants else None, "missed": missed,
                   "detections_on_no_plant": int(matrix[-1, :-1].sum())}
        if taxa is not None:
            levels = level_accuracy(matrix, names, taxa, roles)
            values["species_right"] = levels["species"]["accuracy"]
            if roles is not None:
                values["role_right"] = levels["role"]["accuracy"]
        if decided is not None:
            matrix, names = confusion_matrix(coco, decided, ids, *thresholds)
            values |= decision_summary(decision_matrix(matrix, names, roles))
        return values

    rows = [row("all", None, metrics), *(row(name, ids) for name, ids in sorted(image_ids.items()))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("by dataset:")
    for values in rows:
        parts = [f"{values['images']} images", f"{values['plants']} plants",
                 f"segm AP {_share(values.get('segm_AP'), '.3f')}",
                 f"found {_share(values['found'])}"]
        if "species_right" in values:
            parts.append(f"species right {_share(values['species_right'])}")
        if "harmful_removed" in values:
            parts += [f"harmful removed {_share(values['harmful_removed'])}",
                      f"left {_share(values['harmful_left'])}",
                      f"crops removed {_share(values['crops_removed'])}"]
        print(f"  {values['dataset']}: " + ", ".join(parts))
    print(f"-> {path}")


def _share(value: float | None, spec: str = ".1%") -> str:
    return "-" if value is None else format(value, spec)


def _rules_and_roles(settings: dict, names: list[str]) -> tuple[dict, Any, dict | None]:
    """The EPPO taxa, the decision rules (None without `settings["rules"]`) and the role of
    each class name (None without rules)."""
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    if not settings["rules"]:
        return taxa, None, None
    rules = load_rules(resolve(settings["rules"]), taxa)
    return taxa, rules, {name: role_of(name, rules, taxa) for name in names}


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

