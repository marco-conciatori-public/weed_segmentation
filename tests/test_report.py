import csv

import pytest

from weedseg.evaluation import evaluate_coco, match_detections
from weedseg.report import found_counts, write_by_dataset, write_by_species

H, W = 100, 200
TAXA = {"ZEAMX": {"name": "Zea mays"}, "CHEAL": {"name": "Chenopodium album"}}


def plant(id_, image_id, x, eppo):
    return {"id": id_, "image_id": image_id, "category_id": 1, "bbox": [x, 10, 30, 30],
            "area": 900, "iscrowd": 0, "eppo": eppo}


def det(image_id, x, y=10):
    return {"image_id": image_id, "category_id": 1, "bbox": [x, y, 30, 30], "score": 0.95}


def coco(datasets):
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": W, "height": H, "dataset": d}
                   for i, d in enumerate(datasets, start=1)],
        "categories": [{"id": 1, "name": "plant"}, {"id": 2, "name": "ignore"}],
        # image 1: maize and a goosefoot; image 2: two goosefoots
        "annotations": [plant(1, 1, 10, "ZEAMX"), plant(2, 1, 60, "CHEAL"),
                        plant(3, 2, 10, "CHEAL"), plant(4, 2, 60, "CHEAL")],
    }


# image 1: both found; image 2: one goosefoot found, one missed, one detection on bare soil
RESULTS = [det(1, 10), det(1, 60), det(2, 10), det(2, 120, y=60)]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_by_dataset_splits_the_main_numbers(tmp_path):
    gt = coco(["a", "b"])
    path = tmp_path / "by_dataset.csv"
    write_by_dataset(gt, RESULTS, match_detections(gt, RESULTS), evaluate_coco(gt, RESULTS),
                     100, path)
    rows = {row["dataset"]: row for row in read(path)}
    assert list(rows) == ["all", "a", "b"]
    assert [(r["images"], r["plants"], r["missed"], r["detections_on_no_plant"])
            for r in rows.values()] == [("2", "4", "1", "1"), ("1", "2", "0", "0"),
                                        ("1", "2", "1", "1")]
    assert float(rows["all"]["found"]) == pytest.approx(0.75)
    assert float(rows["a"]["AP"]) == pytest.approx(1.0)
    assert float(rows["b"]["found"]) == pytest.approx(0.5)


def test_a_single_dataset_still_gets_its_row(tmp_path):
    gt = coco(["a", "a"])
    rows = write_by_dataset(gt, RESULTS, match_detections(gt, RESULTS),
                            evaluate_coco(gt, RESULTS), 100, tmp_path / "by_dataset.csv")
    assert [r["dataset"] for r in rows] == ["all", "a"]


def test_by_species_counts_plants_found_per_label(tmp_path):
    gt = coco(["a", "b"])
    rows = write_by_species(gt, match_detections(gt, RESULTS), TAXA, tmp_path / "s.csv")
    assert [(r["eppo"], r["name"], r["plants"], r["found"], r["datasets"]) for r in rows] == [
        ("CHEAL", "Chenopodium album", 3, pytest.approx(2 / 3), "b 2, a 1"),
        ("ZEAMX", "Zea mays", 1, 1.0, "a 1")]
    assert [r["eppo"] for r in read(tmp_path / "s.csv")] == ["CHEAL", "ZEAMX"]


def test_found_counts():
    gt = coco(["a", "b"])
    assert found_counts(match_detections(gt, RESULTS)) == {
        "plants": 4, "found": 3, "missed": 1, "detections_on_no_plant": 1}
