"""Evaluate a trained Mask R-CNN on a split. Settings: configs/maskrcnn.yaml (`evaluate`,
plus `data`, `run_name` and `output_dir` to find the run).

Writes to <run>/eval/<split>_<checkpoint>[_min<N>mm]/ (suffix when data.min_plant_size_mm
is set): metrics.json (COCO bbox and segm), per_class.csv, confusion_matrix.csv/.png,
predictions.json (COCO results) and previews/ (ground truth left, predictions right).
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from weedseg.config import load_config
from weedseg.data.coco import ignore_small_plants
from weedseg.evaluation import confusion_matrix, evaluate_coco
from weedseg.maskrcnn.data import CocoInstanceDataset, collate, load_coco, model_categories
from weedseg.maskrcnn.engine import predict, select_device
from weedseg.maskrcnn.model import load_checkpoint_model
from weedseg.paths import resolve
from weedseg.visualize import draw_annotations, plot_confusion_matrix, put_label

CONFIG = "configs/maskrcnn.yaml"


def main() -> None:
    config = load_config(CONFIG)
    settings = config["evaluate"]
    run_dir = resolve(config["output_dir"]) / config["run_name"]
    checkpoint_path = (run_dir / f"{settings['checkpoint']}.pt"
                       if settings["checkpoint"] in ("best", "last")
                       else resolve(settings["checkpoint"]))
    device = select_device(config["device"])
    amp = config["train"]["amp"] and device.type == "cuda"
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    model_config = checkpoint["config"]["model"]
    print(f"{checkpoint_path} (epoch {checkpoint['epoch']}), device {device}")

    split = settings["split"]
    min_size_mm = config["data"]["min_plant_size_mm"]
    coco = ignore_small_plants(load_coco(resolve(config["data"][split])), min_size_mm)
    # The model's labels follow the training categories; look them up by name in this file.
    by_name = {c["name"]: c for c in model_categories(coco)}
    missing = [c["name"] for c in checkpoint["categories"] if c["name"] not in by_name]
    if missing:
        raise SystemExit(f"Categories of the model missing from the {split} file: {missing}")
    categories = [by_name[c["name"]] for c in checkpoint["categories"]]

    dataset = CocoInstanceDataset(coco, with_targets=False, max_images=settings["max_images"],
                                  seed=config["seed"])
    if settings["max_images"] is not None:  # evaluate and preview only the chosen images
        image_ids = {i["id"] for i in dataset.images}
        coco = {**coco, "images": dataset.images,
                "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids]}
    loader = DataLoader(dataset, settings["batch_size"], collate_fn=collate,
                        num_workers=config["data"]["num_workers"])
    results = predict(model, loader, categories, device, amp, model_config["mask_threshold"])

    size_suffix = "" if min_size_mm is None else f"_min{min_size_mm:g}mm"
    out_dir = run_dir / "eval" / f"{split}_{checkpoint_path.stem}{size_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    metrics = evaluate_coco(coco, results, max_detections=model_config["detections_per_image"],
                            verbose=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_per_class(metrics, out_dir / "per_class.csv")

    matrix, names = confusion_matrix(coco, results, score_threshold=settings["score_threshold"],
                                     iou_threshold=settings["iou_threshold"])
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth \\ predicted", *names])
        writer.writerows([name, *row] for name, row in zip(names, matrix.tolist()))
    plot_confusion_matrix(matrix, names, out_dir / "confusion_matrix.png",
                          f"{config['run_name']}, {split} (score >= {settings['score_threshold']}, "
                          f"IoU >= {settings['iou_threshold']})")

    if settings["num_previews"]:
        save_previews(coco, results, dataset.image_dir, out_dir / "previews",
                      settings["num_previews"], settings["preview_score_threshold"], config["seed"])
    for iou_type, values in metrics.items():
        summary = values["summary"]
        print(f"{iou_type}: AP {summary['AP']:.3f}  AP50 {summary['AP50']:.3f}")
    print(f"-> {out_dir}")


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


if __name__ == "__main__":
    main()
