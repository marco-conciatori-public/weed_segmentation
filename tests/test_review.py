import pytest

from weedseg.data.coco import OTHER_CATEGORY
from weedseg.evaluation import Match
from weedseg.review import classify, crop_window, rank_errors

NAMES = {1: "ZEAMX", 2: "SETVI", 3: "ECHCG", 4: "MATCH", 5: OTHER_CATEGORY}
ROLES = {"ZEAMX": "crop", "SETVI": "harmful", "ECHCG": "harmful", "MATCH": "neutral",
         OTHER_CATEGORY: OTHER_CATEGORY}


def plant(category_id, size=50):
    return {"category_id": category_id, "bbox": [0, 0, size, size]}


def detection(category_id, score):
    return {"category_id": category_id, "score": score, "bbox": [0, 0, 50, 50]}


@pytest.mark.parametrize("true_id, predicted_id, score, expected", [
    (1, 2, 0.9, "crop_removed"),
    (1, 2, 0.6, "other_wrong_class"),  # under defer_below: deferred, not removed
    (2, 1, 0.9, "harmful_kept"),
    (2, 4, 0.9, "harmful_kept"),
    (4, 2, 0.9, "neutral_removed"),
    (5, 3, 0.9, "other_plant_removed"),
    (2, 3, 0.9, "same_role_wrong_species"),
    (1, 4, 0.9, "other_wrong_class"),
    (2, 5, 0.9, "other_wrong_class"),
])
def test_classify_wrong_classes(true_id, predicted_id, score, expected):
    item = classify(Match(1, plant(true_id), detection(predicted_id, score)), NAMES, ROLES, 0.8)
    assert item.error_type == expected


def test_classify_right_class_missed_and_no_plant():
    assert classify(Match(1, plant(2), detection(2, 0.9)), NAMES, ROLES, None) is None
    assert classify(Match(1, plant(2), None), NAMES, ROLES, None).error_type == "missed_harmful"
    assert classify(Match(1, plant(1), None), NAMES, ROLES, None).error_type == "missed_other"
    item = classify(Match(1, None, detection(2, 0.9)), NAMES, ROLES, None)
    assert (item.error_type, item.action) == ("no_plant_detection", "remove")


def test_rank_by_confidence_and_missed_by_size():
    items = [classify(Match(1, plant(1), detection(2, s)), NAMES, ROLES, None) for s in (0.6, 0.9)]
    items += [classify(Match(1, plant(2, size), None), NAMES, ROLES, None) for size in (20, 80)]
    ranked = rank_errors(items)
    assert [item.score for item in ranked["crop_removed"]] == [0.9, 0.6]
    assert [item.box[2] for item in ranked["missed_harmful"]] == [80, 20]
    assert ranked["harmful_kept"] == []  # every type is present, possibly empty


def test_crop_window_has_context_and_stays_inside_the_image():
    assert crop_window([100, 100, 50, 50], 1920, 1088, min_side=400) == (0, 0, 400, 400)
    x0, y0, x1, y1 = crop_window([1800, 1000, 100, 80], 1920, 1088, min_side=400)
    assert (x1, y1) == (1920, 1088) and x1 - x0 == 400 and y1 - y0 == 400
    x0, y0, x1, y1 = crop_window([800, 400, 300, 200], 1920, 1088, min_side=400, context=1.0)
    assert x1 - x0 == 900  # the box's longer side, plus as much again on each side
