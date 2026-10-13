import numpy as np

from weedseg.data.cropandweed import instance_masks, session_of, unboxed_regions


def test_session_of():
    assert session_of("vwg-0328-0004") == "vwg-0328"


def test_instance_masks_split_overlapping_boxes_by_nearest_stem():
    semantic = np.zeros((10, 20), dtype=np.uint8)
    semantic[2:8, 2:18] = 5  # two touching plants of label 5
    semantic[0, 0] = 7  # another label, outside the plants
    boxes = [(2, 2, 11, 7, 5, 4, 4), (8, 2, 17, 7, 5, 15, 4)]  # overlap in columns 8-11

    first, second = instance_masks(semantic, boxes)

    assert not (first & second).any()
    assert ((first | second) == (semantic == 5)).all()
    assert first[4, 9] and second[4, 10]  # column 9 is nearer stem 4, column 10 nearer stem 15


def test_instance_masks_only_take_the_box_label():
    semantic = np.zeros((10, 10), dtype=np.uint8)
    semantic[1:4, 1:4] = 5
    semantic[2, 2] = 6
    (mask,) = instance_masks(semantic, [(0, 0, 5, 5, 5, 2, 2)])
    assert mask.sum() == 8 and not mask[2, 2]


def test_unboxed_regions_become_ignore_instances():
    semantic = np.zeros((40, 40), dtype=np.uint8)
    semantic[0:20, 0:20] = 9  # 400 px, no box
    semantic[30:32, 30:32] = 9  # 4 px, below min_area
    (region,) = unboxed_regions(semantic, [], min_area=256)
    assert region.ignore and region.source_label == "9"
    assert region.bbox_xyxy == (0, 0, 20, 20)
    assert region.mask.sum() == 400
