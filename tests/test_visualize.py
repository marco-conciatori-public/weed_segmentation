import numpy as np

from weedseg.data.coco import encode_mask
from weedseg.visualize import annotation_mask, draw_annotations

CATEGORIES = [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "ignore"}]


def square_mask(x0, y0, x1, y1, shape=(100, 120)):
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_annotation_mask_decodes_rle_and_polygons():
    mask = square_mask(10, 20, 30, 40)
    assert (annotation_mask({"segmentation": encode_mask(mask)}, 100, 120) == mask).all()
    polygon = {"segmentation": [[10, 20, 30, 20, 30, 40, 10, 40]]}
    assert annotation_mask(polygon, 100, 120)[30, 20]
    assert annotation_mask({"bbox": [0, 0, 1, 1]}, 100, 120) is None


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
