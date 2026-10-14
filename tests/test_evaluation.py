import numpy as np
import pytest

from weedseg.data.coco import encode_mask
from weedseg.evaluation import confusion_matrix, evaluate_coco

H, W = 100, 200


def box_mask(x, y, w, h):
    mask = np.zeros((H, W), bool)
    mask[y:y + h, x:x + w] = True
    return encode_mask(mask)


def ann(id_, category_id, x, y, w, h, crowd=0):
    return {"id": id_, "image_id": 1, "category_id": category_id, "bbox": [x, y, w, h],
            "area": w * h, "iscrowd": crowd, "segmentation": box_mask(x, y, w, h)}


def det(category_id, x, y, w, h, score):
    return {"image_id": 1, "category_id": category_id, "bbox": [x, y, w, h], "score": score,
            "segmentation": box_mask(x, y, w, h)}


GT = {
    "images": [{"id": 1, "file_name": "a.jpg", "width": W, "height": H}],
    "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"},
                   {"id": 3, "name": "ignore"}],
    "annotations": [ann(1, 1, 10, 10, 30, 30), ann(2, 2, 60, 10, 30, 30),
                    ann(3, 3, 120, 40, 50, 50, crowd=1)],
}


@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
def test_detection_on_ignore_region_is_not_a_false_positive(iou_type):
    perfect = [det(1, 10, 10, 30, 30, 0.9), det(2, 60, 10, 30, 30, 0.8)]
    on_ignore = [det(2, 125, 45, 40, 40, 0.95)]  # highest score, inside the ignore region
    elsewhere = [det(2, 0, 60, 30, 30, 0.95)]  # highest score, on bare soil
    for results, expected in ((perfect + on_ignore, 1.0), (perfect + elsewhere, None)):
        metrics = evaluate_coco(GT, results, iou_types=(iou_type,))[iou_type]
        weed = next(c for c in metrics["per_class"] if c["category"] == "weed")
        if expected is None:
            assert weed["AP50"] < 1.0
        else:
            assert weed["AP50"] == pytest.approx(expected)
    assert [c["category"] for c in metrics["per_class"]] == ["crop", "weed"]
    assert metrics["per_class"][0]["gt_instances"] == 1


def test_summary_uses_our_max_detections():
    # COCOeval.summarize() would report AP = -1 with any limit but 100.
    perfect = [det(1, 10, 10, 30, 30, 0.9), det(2, 60, 10, 30, 30, 0.8)]
    summary = evaluate_coco(GT, perfect, max_detections=300)["segm"]["summary"]
    assert summary["AP"] == pytest.approx(1.0)
    assert summary["ARmax"] == pytest.approx(1.0)


def test_evaluate_without_detections_gives_zeros():
    metrics = evaluate_coco(GT, [])
    assert metrics["segm"]["summary"]["AP"] == 0.0


def test_confusion_matrix_counts_swaps_misses_and_background():
    results = [
        det(2, 10, 10, 30, 30, 0.9),  # crop predicted as weed
        det(1, 0, 60, 30, 30, 0.9),  # crop on bare soil
        det(1, 125, 45, 40, 40, 0.9),  # on the ignore region: not counted
        det(2, 60, 10, 30, 30, 0.1),  # below the score threshold: the weed counts as missed
    ]
    matrix, names = confusion_matrix(GT, results, score_threshold=0.5)
    assert names == ["crop", "weed", "background"]
    np.testing.assert_array_equal(matrix, [[0, 1, 0],
                                           [0, 0, 1],
                                           [1, 0, 0]])
