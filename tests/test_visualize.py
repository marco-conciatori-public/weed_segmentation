import numpy as np

from weedseg.visualize import PLANT_COLOR, draw_annotations, draw_detections, put_label

CATEGORIES = [{"id": 1, "name": "plant"}, {"id": 2, "name": "ignore"}]


def test_draw_colours_plant_boxes_and_hatches_ignore_regions():
    image = np.zeros((100, 120, 3), np.uint8)
    annotations = [
        {"category_id": 1, "iscrowd": 0, "bbox": [10, 20, 20, 20], "stem": [20, 30],
         "eppo": "ZEAMX"},
        {"category_id": 2, "iscrowd": 1, "bbox": [60, 50, 40, 40]},
    ]
    result = draw_annotations(image, annotations, CATEGORIES, draw_labels=False)
    assert result.shape == image.shape
    assert not result.any(axis=2)[0:5, 0:5].any()  # background untouched
    edge = result[30, 10]
    assert edge.max() > 0 and len(set(edge.tolist())) > 1  # coloured box, not grey
    assert not result[25, 15].any()  # inside the plant's box: untouched
    hatched = result[50:90, 60:100].reshape(-1, 3)
    assert {tuple(p) for p in hatched.tolist()} >= {(60, 60, 60)}
    assert not image.any()  # input not modified


def test_labels_do_not_cover_each_other():
    image = np.zeros((200, 300, 3), np.uint8)
    taken = []
    for _ in range(3):
        put_label(image, "0.91", (20, 50), (0, 200, 0), 0.6, 2, taken)
    assert len(taken) == 3
    for i, a in enumerate(taken):
        for b in taken[i + 1:]:
            assert a[3] <= b[1] or b[3] <= a[1]  # one above the other
    put_label(image, "far away", (250, 190), (0, 200, 0), 0.6, 2, taken)
    assert taken[-1][3] == 190  # nothing to avoid: stays at its corner


def test_draw_detections_draws_each_box():
    image = np.zeros((100, 120, 3), np.uint8)
    plants = [{"box": [10, 20, 20, 20], "plant_score": 0.91},
              {"box": [60, 50, 40, 40], "plant_score": 0.6}]
    result = draw_detections(image, plants, labels=False)
    assert tuple(result[30, 10]) == PLANT_COLOR and tuple(result[70, 60]) == PLANT_COLOR
    assert not result[30, 20].any()  # inside a box: untouched
    assert not image.any()  # input not modified
