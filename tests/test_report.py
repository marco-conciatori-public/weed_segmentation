import csv

import numpy as np
import pytest

from weedseg.data.coco import encode_mask
from weedseg.evaluation import evaluate_coco
from weedseg.report import write_by_dataset

H, W = 100, 200
SETTINGS = {"score_threshold": 0.5, "iou_threshold": 0.5, "match_by": "mask",
            "rules": "configs/rules/example.yaml", "eppo_taxa": "configs/eppo_taxa.csv"}


def box_mask(x, y, w, h):
    mask = np.zeros((H, W), bool)
    mask[y:y + h, x:x + w] = True
    return encode_mask(mask)


def plant(id_, image_id, category_id, x):
    return {"id": id_, "image_id": image_id, "category_id": category_id, "bbox": [x, 10, 30, 30],
            "area": 900, "iscrowd": 0, "segmentation": box_mask(x, 10, 30, 30)}


def det(image_id, category_id, x, y=10):
    return {"image_id": image_id, "category_id": category_id, "bbox": [x, y, 30, 30],
            "score": 0.95, "segmentation": box_mask(x, y, 30, 30),
            "class_scores": {str(category_id): 0.95}}


def coco(datasets):
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": W, "height": H, "dataset": d}
                   for i, d in enumerate(datasets, start=1)],
        "categories": [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "CHEAL"},
                       {"id": 3, "name": "other_plant"}, {"id": 4, "name": "ignore"}],
        # image 1: maize and a goosefoot; image 2: two goosefoots
        "annotations": [plant(1, 1, 1, 10), plant(2, 1, 2, 60), plant(3, 2, 2, 10),
                        plant(4, 2, 2, 60)],
    }


# image 1: both found; image 2: one goosefoot found, one missed, one detection on bare soil
RESULTS = [det(1, 1, 10), det(1, 2, 60), det(2, 2, 10), det(2, 2, 120, y=60)]


def test_by_dataset_splits_the_main_numbers(tmp_path):
    gt = coco(["a", "b"])
    path = tmp_path / "by_dataset.csv"
    write_by_dataset(gt, RESULTS, SETTINGS, evaluate_coco(gt, RESULTS), 100, True, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = {row["dataset"]: row for row in csv.DictReader(f)}
    assert list(rows) == ["all", "a", "b"]
    assert [(r["images"], r["plants"], r["missed"], r["detections_on_no_plant"])
            for r in rows.values()] == [("2", "4", "1", "1"), ("1", "2", "0", "0"),
                                        ("1", "2", "1", "1")]
    assert float(rows["all"]["found"]) == pytest.approx(0.75)
    assert float(rows["a"]["segm_AP"]) == pytest.approx(1.0)
    assert float(rows["b"]["species_right"]) == 1.0
    # maize kept, goosefoots removed unless missed; b has no crop to remove
    assert float(rows["a"]["harmful_removed"]) == 1.0 and rows["a"]["crops_removed"] == "0.0"
    assert float(rows["b"]["harmful_left"]) == 0.5 and rows["b"]["crops_removed"] == ""
    assert rows["b"]["removals_on_no_plant"] == "1"


def test_by_dataset_is_skipped_for_a_single_dataset(tmp_path):
    gt = coco(["a", "a"])
    write_by_dataset(gt, RESULTS, SETTINGS, evaluate_coco(gt, RESULTS), 100, True,
                     tmp_path / "by_dataset.csv")
    assert not (tmp_path / "by_dataset.csv").exists()
