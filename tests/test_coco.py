import numpy as np
import pytest
from pycocotools import mask as mask_utils

from weedseg.data.coco import (
    IGNORE_CATEGORY,
    OTHER_CATEGORY,
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


def test_keep_common_species_applies_the_four_rules(tmp_path):
    (tmp_path / "map.csv").write_text(
        "source_label,role,eppo,species\n"
        "1,crop,ZEAMX,ZEAMX\n"  # common species: class
        "2,weed,POAAN,POAAN\n"  # common species: class
        "3,weed,POATR,POATR\n"  # rare species: other_plant
        "4,weed,1POAG,1POAG\n"  # genus with a class (POAAN): ignore
        "5,weed,1MERG,1MERG\n"  # genus without a class: other_plant
        "255,unknown,,\n",  # unidentified: ignore
        encoding="utf-8",
    )

    def taxon(code, rank, genus, family):
        return {"eppo": code, "rank": rank, "name": code, "genus": genus, "family": family,
                "order": "1ORDO"}

    taxa = {t["eppo"]: t for t in [
        taxon("ZEAMX", "species", "1ZEAG", "1GRAF"), taxon("POAAN", "species", "1POAG", "1GRAF"),
        taxon("POATR", "species", "1POAG", "1GRAF"), taxon("1POAG", "genus", "1POAG", "1GRAF"),
        taxon("1MERG", "genus", "1MERG", "1EUPF")]}
    label_map = LabelMap(tmp_path / "map.csv", "species", taxa)
    images = [{"1", "2", "4"}, {"1", "2", "3"}, {"5", "255"}]
    report = label_map.keep_common_species(images, min_images=2)

    assert label_map.taxonomy == "species_min2"
    assert {label: label_map.category(label) for label in ("1", "2", "3", "4", "5", "255")} == {
        "1": "ZEAMX", "2": "POAAN", "3": OTHER_CATEGORY, "4": IGNORE_CATEGORY,
        "5": OTHER_CATEGORY, "255": IGNORE_CATEGORY}
    categories = label_map.categories()
    assert [c["name"] for c in categories] == ["ZEAMX", "POAAN", OTHER_CATEGORY, IGNORE_CATEGORY]
    assert categories[2]["supercategory"] == "other"
    assert report[0].startswith("2 classes")


def test_keep_common_species_needs_the_species_taxonomy(tmp_path):
    label_map = LabelMap(write_label_map(tmp_path / "map.csv"), "group")
    with pytest.raises(ValueError, match="species"):
        label_map.keep_common_species([], 1)
