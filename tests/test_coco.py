import numpy as np
from pycocotools import mask as mask_utils

from weedseg.data.coco import (
    IGNORE_CATEGORY,
    ImageRecord,
    Instance,
    LabelMap,
    ignore_small_plants,
    split_by_group,
    to_coco,
)


def write_label_map(path):
    path.write_text(
        "source_label,common_name,role,species,group\n"
        "1,Maize,crop,ZEAMX,crop\n"
        "2,White goosefoot,weed,CHEAL,weed\n"
        "3,Vegetation,unknown,,\n",
        encoding="utf-8",
    )
    return path


def test_split_keeps_groups_together_and_follows_fractions():
    items = [f"s{g:03d}-{i}" for g in range(100) for i in range(10)]
    splits = split_by_group(items, lambda name: name.split("-")[0],
                            {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    groups = {name: {item.split("-")[0] for item in split} for name, split in splits.items()}
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert sum(len(split) for split in splits.values()) == len(items)
    assert abs(len(splits["train"]) - 700) <= 10


def test_split_is_reproducible():
    items = [f"s{g}-{i}" for g in range(20) for i in range(3)]
    first = split_by_group(items, lambda n: n.split("-")[0], {"a": 0.5, "b": 0.5}, seed=3)
    second = split_by_group(items, lambda n: n.split("-")[0], {"a": 0.5, "b": 0.5}, seed=3)
    assert first == second


def test_label_map_categories_and_ignore(tmp_path):
    label_map = LabelMap(write_label_map(tmp_path / "map.csv"), "species")
    names = [c["name"] for c in label_map.categories()]
    assert names == ["ZEAMX", "CHEAL", IGNORE_CATEGORY]  # crops first, ignore last
    assert label_map.category("3") == IGNORE_CATEGORY

    grouped = LabelMap(tmp_path / "map.csv", "group")
    assert [c["name"] for c in grouped.categories()] == ["crop", "weed", IGNORE_CATEGORY]


def test_to_coco_writes_masks_stems_and_crowd_flags(tmp_path):
    label_map = LabelMap(write_label_map(tmp_path / "map.csv"), "species")
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 3:8] = True
    record = ImageRecord(
        file_name="a.jpg", width=20, height=10, group="g1",
        instances=[
            Instance("1", (3, 2, 8, 5), mask=mask, stem_xy=(5, 3)),
            Instance("3", (0, 0, 2, 2)),
            Instance("2", (10, 0, 12, 2), ignore=True),
        ],
        attributes={"session": "g1"},
    )
    coco = to_coco([record], label_map, {"split": "train"})

    first, vegetation, ignored = coco["annotations"]
    assert first["bbox"] == [3, 2, 5, 3]
    assert first["area"] == 15
    assert first["stem"] == [5, 3]
    assert first["iscrowd"] == 0
    assert (mask_utils.decode(first["segmentation"]).astype(bool) == mask).all()
    assert vegetation["iscrowd"] == 1 and "segmentation" not in vegetation
    assert ignored["iscrowd"] == 1
    assert coco["images"][0]["session"] == "g1"


def test_ignore_small_plants_uses_mm_and_ignore_category():
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "mm_per_px": 0.5}],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": IGNORE_CATEGORY}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 30, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 10], "iscrowd": 0},
        ],
    }
    result = ignore_small_plants(coco, 20)  # 15 mm and 25 mm
    small, large = result["annotations"]
    assert (small["category_id"], small["iscrowd"], small["ignored_category_id"]) == (2, 1, 1)
    assert large == coco["annotations"][1]
    assert coco["annotations"][0]["category_id"] == 1  # input untouched
    assert ignore_small_plants(coco, None) is coco
