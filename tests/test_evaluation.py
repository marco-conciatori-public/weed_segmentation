import pytest

from weedseg.evaluation import evaluate_coco, match_detections

H, W = 100, 200


def ann(id_, category_id, x, y, w, h, crowd=0):
    return {"id": id_, "image_id": 1, "category_id": category_id, "bbox": [x, y, w, h],
            "area": w * h, "iscrowd": crowd}


def det(x, y, w, h, score, category_id=1):
    return {"image_id": 1, "category_id": category_id, "bbox": [x, y, w, h], "score": score}


GT = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}],
    "categories": [{"id": 1, "name": "plant"}, {"id": 2, "name": "ignore"}],
    "annotations": [ann(1, 1, 10, 10, 30, 30), ann(2, 1, 60, 10, 30, 30),
                    ann(3, 2, 120, 40, 50, 50, crowd=1)],
}
PERFECT = [det(10, 10, 30, 30, 0.9), det(60, 10, 30, 30, 0.8)]


def test_detection_on_ignore_region_is_not_a_false_positive():
    on_ignore = [det(125, 45, 40, 40, 0.95)]  # highest score, inside the ignore region
    elsewhere = [det(0, 60, 30, 30, 0.95)]  # highest score, on bare soil
    assert evaluate_coco(GT, PERFECT + on_ignore)["summary"]["AP50"] == pytest.approx(1.0)
    assert evaluate_coco(GT, PERFECT + elsewhere)["summary"]["AP50"] < 1.0
    per_class = evaluate_coco(GT, PERFECT)["per_class"]
    assert [(c["category"], c["gt_instances"]) for c in per_class] == [("plant", 2)]


def test_crowd_region_of_another_category_is_ignored_too():
    # An ignore region that kept a real category must also cover the plant's detections.
    gt = {**GT, "categories": GT["categories"] + [{"id": 3, "name": "weed"}],
          "annotations": GT["annotations"][:2] + [ann(3, 3, 120, 40, 50, 50, crowd=1)]}
    results = PERFECT + [det(125, 45, 40, 40, 0.95)]
    plant = next(c for c in evaluate_coco(gt, results)["per_class"] if c["category"] == "plant")
    assert plant["AP50"] == pytest.approx(1.0)


def test_summary_uses_our_max_detections():
    # COCOeval.summarize() would report AP = -1 with any limit but 100.
    summary = evaluate_coco(GT, PERFECT, max_detections=300)["summary"]
    assert summary["AP"] == pytest.approx(1.0)
    assert summary["ARmax"] == pytest.approx(1.0)


def test_evaluate_without_detections_gives_zeros():
    assert evaluate_coco(GT, [])["summary"]["AP"] == 0.0


def test_matching_counts_found_missed_and_detections_on_no_plant():
    results = [
        det(12, 12, 30, 30, 0.9),  # the first plant, IoU ~0.78
        det(0, 60, 30, 30, 0.9),  # bare soil
        det(125, 45, 40, 40, 0.9),  # on the ignore region: dropped
        det(60, 10, 30, 30, 0.1),  # below the score threshold: the second plant is missed
    ]
    matches = match_detections(GT, results, score_threshold=0.5, iou_threshold=0.5)
    found = [(m.plant["id"], m.detection["bbox"]) for m in matches if m.plant and m.detection]
    assert found == [(1, [12, 12, 30, 30])]
    assert [m.detection["bbox"] for m in matches if m.plant is None] == [[0, 60, 30, 30]]
    assert [m.plant["id"] for m in matches if m.detection is None] == [2]


def test_each_plant_and_detection_is_matched_once_best_score_first():
    results = [det(10, 10, 30, 30, 0.6), det(11, 11, 30, 30, 0.9)]  # two on the first plant
    matches = match_detections(GT, results)
    pairs = [(m.plant and m.plant["id"], m.detection and m.detection["score"]) for m in matches]
    assert sorted(pairs, key=str) == sorted([(1, 0.9), (None, 0.6), (2, None)], key=str)
