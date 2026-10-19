"""The model's worst errors, as images plus a CSV form, to show a plant expert.

Settings: configs/maskrcnn.yaml (`review`, plus the `evaluate` settings that locate the
evaluation and set the thresholds and decision rules). Run scripts/evaluate_maskrcnn.py first:
this script reads its predictions.json, so it needs no GPU.

Writes <run>/eval/<split>_<checkpoint>/review/ (replaced on each run): README.txt,
summary.csv (the number of errors of each type) and one folder per error type (see
weedseg.review.ERROR_TYPES) with the top `per_type` errors as images (labels left, model
right, the plant in question framed in white) and review.csv, a form for the expert's verdicts.
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from weedseg.config import load_config
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import load_rules, role_of
from weedseg.eppo import NAME_SEPARATOR, load_taxa
from weedseg.evaluation import match_detections
from weedseg.maskrcnn.data import CocoInstanceDataset, load_coco
from weedseg.maskrcnn.run import eval_dir, split_file
from weedseg.paths import resolve
from weedseg.review import ERROR_TYPES, ReviewItem, classify, crop_window, rank_errors
from weedseg.visualize import draw_annotations

CONFIG = "configs/maskrcnn.yaml"
FRAME = (255, 255, 255)  # BGR frame around the plant in question
README = """Errors of the weed recognition model, for review.

Each folder holds one type of error, the most confident errors first (for plants the model
missed: the largest first). Each image shows the same spot twice: on the left the annotators'
labels, on the right the model's predictions with their confidence (0-1). The plant in
question is framed in white. Grey hatched areas were left out of training and scoring
(unidentified plants, or plants under {min_size}).

Please fill in the two last columns of each folder's review.csv:
- verdict: "label wrong" (the annotators made a mistake), "model wrong", or "unclear".
  For no_plant_detection: "plant" (the annotators missed it; please name the species if you
  can) or "not a plant".
- comment: anything useful, e.g. the right species, or why the error is understandable.

Roles come from the decision rules {rules}: crop and neutral plants are kept, harmful weeds
removed, and other_plant (species the model doesn't know) handed to a generic model, as is
any detection less confident than {defer_below}.

Folders:
{types}
"""
CSV_COLUMNS = ["rank", "image", "source_image", "labelled_as", "predicted_as", "confidence",
               "true_role", "robot_action", "plant_size_mm", "box_x", "box_y", "box_w", "box_h",
               "verdict", "comment"]


def main() -> None:
    config = load_config(CONFIG)
    settings, review = config["evaluate"], config["review"]
    if not settings["rules"]:
        raise SystemExit("review_errors.py needs decision rules (evaluate.rules)")
    source = eval_dir(config)
    if not (source / "predictions.json").exists():
        raise SystemExit(f"No predictions in {source}; run scripts/evaluate_maskrcnn.py first")
    with open(source / "predictions.json", encoding="utf-8") as f:
        results = json.load(f)

    coco = load_coco(split_file(config, settings["split"]))
    # Evaluation may have used a random subset; review the same images.
    images = CocoInstanceDataset(coco, with_targets=False, max_images=settings["max_images"],
                                 seed=config["seed"]).images
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    rules = load_rules(resolve(settings["rules"]), taxa)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    roles = {name: role_of(name, rules, taxa) for name in names.values() if name != "ignore"}
    species_of_label = read_label_species(review["label_map"])

    matches = match_detections(coco, results, [i["id"] for i in images],
                               settings["score_threshold"], settings["iou_threshold"],
                               settings["match_by"])
    items = [item for match in matches
             if (item := classify(match, names, roles, rules.defer_below)) is not None]
    ranked = rank_errors(items)

    out_dir = source / "review"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    min_size = coco["info"].get("min_plant_size_mm")
    (out_dir / "README.txt").write_text(README.format(
        min_size=f"{min_size:g} mm" if min_size else "the minimum size",
        rules=Path(settings["rules"]).name,
        defer_below=rules.defer_below if rules.defer_below is not None else "(not set)",
        types="\n".join(f"- {name}: {text}" for name, text in ERROR_TYPES)), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_type", "count", "description"])
        writer.writerows([name, len(ranked[name]), text] for name, text in ERROR_TYPES)

    by_id = {i["id"]: i for i in coco["images"]}
    annotations, detections = defaultdict(list), defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    for result in results:
        if result["score"] >= settings["score_threshold"]:
            detections[result["image_id"]].append(result)
    display = [{**c, "name": short_name(c["name"], taxa)} for c in coco["categories"]]
    image_dir = resolve(coco["info"]["image_dir"])

    for error_type, _ in ERROR_TYPES:
        folder = out_dir / error_type
        folder.mkdir()
        rows = []
        for rank, item in enumerate(ranked[error_type][:review["per_type"]], start=1):
            info = by_id[item.match.image_id]
            file_name = f"{rank:03d}_{Path(info['file_name']).stem}.jpg"
            image = cv2.imread(str(image_dir / info["file_name"]))
            panel = render(image, item, annotations[info["id"]], detections[info["id"]],
                           display, review["min_crop_px"])
            cv2.imwrite(str(folder / file_name), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
            x, y, w, h = item.box
            plant = item.match.plant
            rows.append({
                "rank": rank, "image": file_name, "source_image": info["file_name"],
                "labelled_as": (full_name(item.true_name, taxa, plant, species_of_label)
                                if plant else "(no plant labelled)"),
                "predicted_as": (full_name(item.predicted_name, taxa)
                                 if item.predicted_name else "(not detected)"),
                "confidence": f"{item.score:.3f}" if item.score is not None else "",
                "true_role": roles[item.true_name] if item.true_name else "",
                "robot_action": item.action or "(none: not detected)",
                "plant_size_mm": (f"{max(w, h) * info['mm_per_px']:.0f}"
                                  if info.get("mm_per_px") else ""),
                "box_x": round(x), "box_y": round(y), "box_w": round(w), "box_h": round(h),
                "verdict": "", "comment": "",
            })
        with open(folder / "review.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{error_type}: {len(ranked[error_type])} errors, {len(rows)} written")
    print(f"-> {out_dir}")


def read_label_species(path: str | None) -> dict[str, str]:
    """Source label -> EPPO code, from a dataset's label map (to name other_plant plants)."""
    if not path:
        return {}
    with open(resolve(path), newline="", encoding="utf-8") as f:
        return {row["source_label"]: row["eppo"] for row in csv.DictReader(f)}


def short_name(name: str, taxa: dict[str, dict[str, str]]) -> str:
    """Scientific name for the images (other_plant and unknown names stay as they are)."""
    return taxa[name]["name"] if name in taxa else name


def full_name(name: str, taxa: dict[str, dict[str, str]], plant: dict | None = None,
              species_of_label: dict[str, str] | None = None) -> str:
    """E.g. "Zea mays (maize; EPPO ZEAMX)"; for an other_plant plant, also its real species
    when the label map tells it."""
    if name == OTHER_CATEGORY:
        code = (species_of_label or {}).get(str((plant or {}).get("source_label")), "")
        return f"{OTHER_CATEGORY}: {full_name(code, taxa)}" if code in taxa else OTHER_CATEGORY
    row = taxa[name]
    common = row["common_names_en"].split(NAME_SEPARATOR)[0]
    return f"{row['name']} ({common + '; ' if common else ''}EPPO {name})"


def render(image: np.ndarray, item: ReviewItem, annotations: list[dict], detections: list[dict],
           categories: list[dict], min_crop_px: int) -> np.ndarray:
    """Labels left, predictions right, cropped around the plant in question."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = crop_window(item.box, width, height, min_crop_px)
    panels = []
    for drawn in (draw_annotations(image, annotations, categories, draw_stems=False),
                  draw_annotations(image, detections, categories, draw_stems=False)):
        x, y, w, h = (round(v) for v in item.box)
        cv2.rectangle(drawn, (x - 6, y - 6), (x + w + 6, y + h + 6), FRAME, 3)
        panels.append(drawn[y0:y1, x0:x1])
    gap = np.full((y1 - y0, 8, 3), 255, np.uint8)
    return np.hstack([panels[0], gap, panels[1]])


if __name__ == "__main__":
    main()
