import json

import cv2
import numpy as np

from weedseg.data import phenobench
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

# Two plants per image: a sugar beet (semantic 1) and a partly visible weed (4) at the border.
PLANTS = [(1, 1, (2, 3, 10, 8)), (2, 4, (0, 12, 3, 16))]  # id, semantic label, x0 y0 x1 y1


def write_image(raw, split, stem, plants=PLANTS):
    semantic = np.zeros((20, 30), np.uint16)
    instances = np.zeros((20, 30), np.uint16)
    for plant_id, label, (x0, y0, x1, y1) in plants:
        semantic[y0:y1, x0:x1] = label
        instances[y0:y1, x0:x1] = plant_id
    for folder, image in (("semantics", semantic), ("plant_instances", instances)):
        (raw / split / folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw / split / folder / f"{stem}.png"), image)


def write_dataset(tmp_path):
    raw = tmp_path / "PhenoBench"
    write_image(raw, "train", "05-15_00028_P0030852")
    write_image(raw, "train", "06-05_00029_P0031000")
    write_image(raw, "val", "05-26_00031_P0030900", plants=[])
    return raw


def test_names_photos_split_and_boxes(tmp_path):
    raw = write_dataset(tmp_path)
    names = phenobench.image_names(raw)
    assert names == ["train/05-15_00028_P0030852", "train/06-05_00029_P0031000",
                     "val/05-26_00031_P0030900"]
    assert phenobench.photo_of(names[0]) == "05-15_P0030852"
    assert phenobench.own_splits(names) == {"train": names[:2], "val": names[2:], "test": []}
    assert phenobench.box_labels(raw, names[0]) == [("crop", 8, 5), ("partial_weed", 3, 4)]


def test_records_ignore_partly_visible_plants(tmp_path):
    raw = write_dataset(tmp_path)
    (record,) = phenobench.iter_records(raw, ["train/05-15_00028_P0030852"], 1.0)
    assert (record.file_name, record.width, record.height) == (
        "train/images/05-15_00028_P0030852.png", 30, 20)
    assert record.group == "05-15_P0030852"
    assert record.attributes == {"date": "2020-05-15", "photo": "P0030852", "mm_per_px": 1.0,
                                 "mm_per_px_estimated": False}
    beet, weed = record.instances
    assert beet.bbox_xyxy == (2, 3, 10, 8) and beet.mask.sum() == 40 and not beet.ignore
    assert weed.source_label == "partial_weed" and weed.ignore
    (kept,) = phenobench.iter_records(raw, ["train/05-15_00028_P0030852"], ignore_partial=False)
    assert not kept.instances[1].ignore and "mm_per_px" not in kept.attributes


def test_conversion_keeps_the_own_split(tmp_path):
    raw = write_dataset(tmp_path)
    names = phenobench.image_names(raw)
    label_map = LabelMap(resolve("configs/label_maps/phenobench.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    settings = {"output_dir": str(tmp_path / "out"), "min_plant_size_mm": 2,
                "mm_per_px_estimate": 1.0, "keep_negative_images": True}
    out = convert_dataset(names, phenobench.photo_of,
                          lambda name: phenobench.box_labels(raw, name),
                          lambda split_names: phenobench.iter_records(raw, split_names, 1.0),
                          label_map, settings, {"name": "PhenoBench"},
                          splits=phenobench.own_splits(names))
    files = {split: json.loads((out / f"{split}.json").read_text(encoding="utf-8"))
             for split in ("train", "val", "test")}
    assert [len(files[split]["images"]) for split in ("train", "val", "test")] == [2, 1, 0]
    assert files["train"]["info"]["split_seed"] is None
    names_of = {c["id"]: c["name"] for c in files["train"]["categories"]}
    assert [(names_of[a["category_id"]], a["iscrowd"]) for a in files["train"]["annotations"]
            ][:2] == [("BEAVA", 0), ("1PLAK", 1)]  # the partial weed keeps its code, ignored
