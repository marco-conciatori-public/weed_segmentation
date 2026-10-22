import json

from weedseg.data import oppd
from weedseg.data.coco import BOX_ONLY, LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

# A charlock tray: a charlock, an unidentified dicot and a rectangle without extent.
OBJECTS = [("SINAR", [[2, 3], [9, 7]]), ("PPPDD", [[0, 12], [2, 15]]), ("STEME", [[0, 0], [0, 0]])]


def write_image(raw, name, sown="SINAR", objects=OBJECTS):
    trial, box, _ = name.split("_", 2)
    tags = {"eppo": sown, "latin": "Sinapis arvensis", "date": "2017-05-19T13:38:59.139000",
            "trial_id": trial, "box_id": box}
    annotation = {
        "description": "",
        "tags": [{"name": key, "value": value} for key, value in tags.items()],
        "size": {"height": 20, "width": 30},
        "objects": [{"classTitle": label, "geometryType": "rectangle",
                     "points": {"exterior": corners, "interior": []}}
                    for label, corners in objects],
    }
    (raw / "ds0" / "ann").mkdir(parents=True, exist_ok=True)
    (raw / "ds0" / "ann" / f"{name}.jpg.json").write_text(json.dumps(annotation),
                                                         encoding="utf-8")


def write_dataset(tmp_path):
    raw = tmp_path / "oppd"
    write_image(raw, "T01_Box053_2017-05-19T13-38-59-139")
    write_image(raw, "T01_Box053_2017-05-20T09-01-02-003")
    write_image(raw, "T02_Box053_2017-09-04T10-00-00-000", sown="STEME", objects=[])
    return raw


def test_names_trays_and_boxes(tmp_path):
    raw = write_dataset(tmp_path)
    names = oppd.image_names(raw)
    assert names == ["T01_Box053_2017-05-19T13-38-59-139", "T01_Box053_2017-05-20T09-01-02-003",
                     "T02_Box053_2017-09-04T10-00-00-000"]
    # Box ids repeat across trials: the trial is part of the tray.
    assert [oppd.tray_of(name) for name in names] == ["T01_Box053", "T01_Box053", "T02_Box053"]
    # Both corners inside the box; the rectangle without extent is dropped.
    assert oppd.box_labels(raw, names[0]) == [("SINAR", 8, 5), ("PPPDD", 3, 4)]


def test_records(tmp_path):
    raw = write_dataset(tmp_path)
    (record,) = oppd.iter_records(raw, ["T01_Box053_2017-05-19T13-38-59-139"], 0.15)
    assert (record.file_name, record.width, record.height, record.group) == (
        "T01_Box053_2017-05-19T13-38-59-139.jpg", 30, 20, "T01_Box053")
    assert record.attributes == {"trial": "T01", "tray": "Box053", "date": "2017-05-19",
                                 "sown_species": "SINAR", "mm_per_px": 0.15,
                                 "mm_per_px_estimated": False}
    charlock, dicot = record.instances
    assert charlock.bbox_xyxy == (2, 3, 10, 8) and charlock.mask is None
    assert dicot.source_label == "PPPDD" and not dicot.ignore


def test_conversion_marks_box_only_and_maps_unidentified_plants(tmp_path):
    raw = write_dataset(tmp_path)
    names = oppd.image_names(raw)
    label_map = LabelMap(resolve("configs/label_maps/oppd.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    # Plants not identified to species are 1PLAK; "Unclassified" is no plant taxon.
    assert label_map.category("PPPDD") == label_map.category("PPPMM") == "1PLAK"
    assert label_map.category("3UNCLK") == "ignore"
    settings = {"output_dir": str(tmp_path / "out"), "min_plant_size_mm": 3,
                "mm_per_px_estimate": 0.5, "keep_negative_images": False,
                "split": {"seed": 0, "fractions": {"train": 0.5, "val": 0.5}}}
    out = convert_dataset(names, oppd.tray_of, lambda name: oppd.box_labels(raw, name),
                          lambda split_names: oppd.iter_records(raw, split_names, 0.5),
                          label_map, settings, {"name": "OPPD"}, box_only=True)
    files = [json.loads((out / f"{split}.json").read_text(encoding="utf-8"))
             for split in ("train", "val")]
    images = [image for coco in files for image in coco["images"]]
    # The tray without plants is dropped; the other tray's two images stay together.
    assert sorted(len(coco["images"]) for coco in files) == [0, 2]
    assert all(image[BOX_ONLY] and image["group"] == "T01_Box053" for image in images)
    coco = next(coco for coco in files if coco["images"])
    names_of = {c["id"]: c["name"] for c in coco["categories"]}
    # The dicot (4 px, 2 mm at 0.5 mm/px) is under 3 mm: an ignore region.
    assert [(names_of[a["category_id"]], a["iscrowd"], a["source_label"])
            for a in coco["annotations"]][:2] == [("SINAR", 0, "SINAR"), ("ignore", 1, "PPPDD")]
    assert "segmentation" not in coco["annotations"][0]
