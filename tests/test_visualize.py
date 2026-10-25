import numpy as np

from weedseg.data.coco import decode_mask, encode_mask
from weedseg.visualize import draw_annotations, draw_plants, put_label

CATEGORIES = [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "ignore"}]


def square_mask(x0, y0, x1, y1, shape=(100, 120)):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_decode_mask_reads_rle_and_polygons():
    mask = square_mask(10, 20, 30, 40)
    assert (decode_mask({"segmentation": encode_mask(mask)}, 100, 120) == mask).all()
    polygon = {"segmentation": [[10, 20, 30, 20, 30, 40, 10, 40]]}
    assert decode_mask(polygon, 100, 120)[30, 20]
    assert decode_mask({"bbox": [0, 0, 1, 1]}, 100, 120) is None


def test_draw_colours_instances_and_greys_ignore_regions():
    image = np.zeros((100, 120, 3), np.uint8)
    annotations = [
        {"category_id": 1, "iscrowd": 0, "bbox": [10, 20, 20, 20],
         "segmentation": encode_mask(square_mask(10, 20, 30, 40)), "stem": [20, 30]},
        {"category_id": 2, "iscrowd": 1, "bbox": [60, 50, 40, 40],
         "segmentation": encode_mask(square_mask(60, 50, 100, 90))},
    ]
    result = draw_annotations(image, annotations, CATEGORIES, draw_labels=False)
    assert result.shape == image.shape
    assert not result.any(axis=2)[0:5, 0:5].any()  # background untouched
    inside = result[25, 12]
    assert inside.max() > 0 and len(set(inside.tolist())) > 1  # coloured, not grey
    ignore = result[70, 80]
    assert len(set(ignore.tolist())) == 1  # grey (or the darker hatch)
    assert not image.any()  # input not modified


def test_labels_do_not_cover_each_other():
    image = np.zeros((200, 300, 3), np.uint8)
    taken = []
    for _ in range(3):
        put_label(image, "ZEAMX 0.91", (20, 50), (0, 200, 0), 0.6, 2, taken)
    assert len(taken) == 3
    for i, a in enumerate(taken):
        for b in taken[i + 1:]:
            assert a[3] <= b[1] or b[3] <= a[1]  # one above the other
    put_label(image, "far away", (250, 190), (0, 200, 0), 0.6, 2, taken)
    assert taken[-1][3] == 190  # nothing to avoid: stays at its corner


def test_draw_plants_marks_zero_shot_plants_white():
    image = np.zeros((100, 120, 3), np.uint8)
    square = [[10, 20], [30, 20], [30, 40], [10, 40]]
    plants = [
        {"box": [10, 20, 20, 20], "polygons": [square], "mask": square_mask(10, 20, 31, 41),
         "species": "ZEAMX", "species_name": "Zea mays", "species_probability": 0.91,
         "send_to_zero_shot": False},
        {"box": [60, 50, 40, 40], "polygons": [[[60, 50], [100, 50], [100, 90], [60, 90]]],
         "species": "other_plant", "species_name": "Other plant", "species_probability": 0.4,
         "send_to_zero_shot": True},
    ]
    result = draw_plants(image, plants, ["ZEAMX", "other_plant"], label="name", mask_alpha=0.5)
    inside = result[35, 20]
    assert len(set(inside.tolist())) > 1  # species colour
    assert len(set(result[80, 80].tolist())) == 1 and result[80, 80].max() > 0  # white fill
    assert not image.any()  # input not modified
