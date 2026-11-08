"""Evaluation outputs of the plant detector's predictions (COCO results) on a split, written
by scripts/evaluate_detector.py.

`write_evaluation` writes to one folder: predictions.json, metrics.json (COCO box metrics),
by_dataset.csv (AP and plants found per source dataset, and for all), by_species.csv (plants
found per EPPO code of their label), previews/ (ground truth left, predictions right) and
evaluation.json (what was evaluated, with which settings).
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from weedseg.eppo import load_taxa
from weedseg.evaluation import Match, evaluate_coco, match_detections
from weedseg.paths import project_root, resolve
from weedseg.visualize import draw_annotations, put_label

ALL = "all"


def write_evaluation(coco: dict[str, Any], results: list[dict[str, Any]],
                     settings: dict[str, Any], out_dir: Path, max_detections: int,
                     image_dir: Path, split_file: Path, seed: int) -> dict[str, Any]:
    """Write every evaluation output of `results` on `coco` (already restricted to the
    evaluated images) to `out_dir`, and return the COCO metrics of all images.

    `settings` is the `evaluate` config section: score_threshold, iou_threshold, eppo_taxa,
    num_previews, preview_score_threshold, split and max_images.
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

    matches = match_detections(coco, results, score_threshold=settings["score_threshold"],
                               iou_threshold=settings["iou_threshold"])
    write_by_dataset(coco, results, matches, metrics, max_detections, out_dir / "by_dataset.csv")
    write_by_species(coco, matches, load_taxa(resolve(settings["eppo_taxa"])),
                     out_dir / "by_species.csv")
    if settings["num_previews"]:
        save_previews(coco, results, image_dir, out_dir / "previews", settings["num_previews"],
                      settings["preview_score_threshold"], seed)
    print(f"box AP {metrics['summary']['AP']:.3f}  AP50 {metrics['summary']['AP50']:.3f}")
    print(f"-> {out_dir}")
    return metrics


def _relative(path: Path) -> str:
    """`path` relative to the project root when inside it (portable between machines)."""
    try:
        return Path(path).resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def found_counts(matches: list[Match]) -> dict[str, int]:
    """Labelled plants, those found and missed, and detections on no labelled plant."""
    plants = sum(m.plant is not None for m in matches)
    missed = sum(m.plant is not None and m.detection is None for m in matches)
    return {"plants": plants, "found": plants - missed, "missed": missed,
            "detections_on_no_plant": sum(m.plant is None for m in matches)}


def write_by_dataset(coco: dict, results: list[dict], matches: list[Match], metrics: dict,
                     max_detections: int, path: Path) -> list[dict[str, Any]]:
    """One row for all images, then one per source dataset (the images' `dataset` field): COCO
    box AP and AP50, and the labelled plants found at the score and IoU thresholds of
    `matches`. Shows how the detector does on each dataset, which the overall numbers can hide
    (a dense dataset weighs most in them). `metrics` are those of all images (evaluate_coco),
    so they aren't computed twice. The rows."""
    image_ids = defaultdict(list)
    for image in coco["images"]:
        image_ids[image.get("dataset") or "(none)"].append(image["id"])
    matches_of = defaultdict(list)
    dataset_of = {image["id"]: image.get("dataset") or "(none)" for image in coco["images"]}
    for match in matches:
        matches_of[dataset_of[match.image_id]].append(match)

    def row(name: str, ids: list[int], subset: list[Match], metrics: dict | None) -> dict:
        if metrics is None:
            metrics = evaluate_coco(coco, results, ids, max_detections=max_detections)
        counts = found_counts(subset)
        return {"dataset": name, "images": len(ids), "plants": counts["plants"],
                "AP": metrics["summary"]["AP"], "AP50": metrics["summary"]["AP50"],
                "found": counts["found"] / counts["plants"] if counts["plants"] else None,
                "missed": counts["missed"],
                "detections_on_no_plant": counts["detections_on_no_plant"]}

    rows = [row(ALL, [i["id"] for i in coco["images"]], matches, metrics),
            *(row(name, ids, matches_of[name], None) for name, ids in sorted(image_ids.items()))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("by dataset:")
    for r in rows:
        print(f"  {r['dataset']}: {r['images']} images, {r['plants']} plants, "
              f"AP {r['AP']:.3f}, AP50 {r['AP50']:.3f}, found {_share(r['found'])}, "
              f"{r['detections_on_no_plant']} detections on no labelled plant")
    print(f"-> {path}")
    return rows


def write_by_species(coco: dict, matches: list[Match], taxa: dict[str, dict[str, str]],
                     path: Path) -> list[dict[str, Any]]:
    """One row per EPPO code of the labelled plants' own labels (the `eppo` field that
    combining keeps): the plants and the share found, most plants first, with the datasets
    they come from. Shows which plants the detector misses (e.g. thin grasses). The rows."""
    dataset_of = {image["id"]: image.get("dataset") or "(none)" for image in coco["images"]}
    plants, found, datasets = Counter(), Counter(), defaultdict(Counter)
    for match in matches:
        if match.plant is None:
            continue
        code = match.plant.get("eppo") or "(none)"
        plants[code] += 1
        found[code] += match.detection is not None
        datasets[code][dataset_of[match.image_id]] += 1
    rows = [{"eppo": code, "name": taxa.get(code, {}).get("name", ""), "plants": n,
             "found": found[code] / n,
             "datasets": ", ".join(f"{d} {k}" for d, k in datasets[code].most_common())}
            for code, n in sorted(plants.items(), key=lambda kv: (-kv[1], kv[0]))]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["eppo", "name", "plants", "found", "datasets"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {path}")
    return rows


def _share(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


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
