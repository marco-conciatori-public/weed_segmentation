import json

import pytest

from weedseg.data import cornweed
from weedseg.data.coco import LabelMap
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

BY_WIDTH = {1280: 1.07, 640: 1.3}


def write_dataset(tmp_path):
    root = tmp_path / cornweed.ROOT / "coco_annotations"
    root.mkdir(parents=True)
    categories = [{"id": 1, "name": "Weeds"}, {"id": 2, "name": "Maize"}]
    train = {"categories": categories,
             "images": [{"id": 0, "file_name": "Langsenkamp_210727_color_00773.png",
                         "width": 1280, "height": 720},
                        {"id": 1, "file_name": "Langsenkamp_210722_bonirob_color_04724.png",
                         "width": 640, "height": 480}],
             "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [10, 20, 30, 40]},
                             {"id": 1, "image_id": 0, "category_id": 2, "bbox": [100, 50, 80, 90]},
                             {"id": 2, "image_id": 1, "category_id": 1, "bbox": [5, 5, 10, 12]}]}
    valid = {"categories": categories,
             "images": [{"id": 9, "file_name": "Langsenkamp_210727_color_00790.png",
                         "width": 1280, "height": 720}],
             "annotations": []}
    for name, coco in (("train_2022.json", train), ("valid_2022.json", valid)):
        (root / name).write_text(json.dumps(coco), encoding="utf-8")
    return tmp_path


def test_names_runs_and_groups(tmp_path):
    raw = write_dataset(tmp_path)
    assert cornweed.image_names(raw) == ["Langsenkamp_210722_bonirob_color_04724.png",
                                         "Langsenkamp_210727_color_00773.png",
                                         "Langsenkamp_210727_color_00790.png"]
    info = cornweed.parse("Langsenkamp_210722_bonirob_color_04724.png")
    assert info == {"run": "Langsenkamp_210722_bonirob_color", "field": "Langsenkamp",
                    "date": "2021-07-22", "platform": "bonirob", "frame": 4724}
    assert cornweed.parse("Langsenkamp_210727_color_00773.png")["platform"] == "tractor"
    # neighbouring frames of one run share a group; the same frame block of another run doesn't
    assert (cornweed.group_of("Langsenkamp_210727_color_00773.png", 50)
            == cornweed.group_of("Langsenkamp_210727_color_00790.png", 50)
            == "Langsenkamp_210727_color_00750")
    with pytest.raises(ValueError, match="Unexpected"):
        cornweed.parse("something_else.png")


def test_records_boxes_and_scale(tmp_path):
    raw = write_dataset(tmp_path)
    names = cornweed.image_names(raw)
    assert cornweed.box_labels(raw, names[1]) == [("Weeds", 30, 40), ("Maize", 80, 90)]
    small, tractor, empty = cornweed.iter_records(raw, names, 50, BY_WIDTH)
    assert (tractor.file_name, tractor.width, tractor.height) == (names[1], 1280, 720)
    assert [(i.source_label, i.bbox_xyxy) for i in tractor.instances] == [
        ("Weeds", (10, 20, 40, 60)), ("Maize", (100, 50, 180, 140))]
    assert tractor.attributes["mm_per_px"] == 1.07 and small.attributes["mm_per_px"] == 1.3
    assert tractor.attributes["date"] == "2021-07-27" and not empty.instances
    with pytest.raises(ValueError, match="mm_per_px_by_width"):
        cornweed.mm_per_px(raw, names[0], {1280: 1.07})


def test_thin_frames_keeps_one_in_every_stride():
    names = [f"Langsenkamp_210727_color_{frame:05d}.png" for frame in range(770, 776)]
    assert cornweed.thin_frames(names, 1) == names
    assert cornweed.thin_frames(names, 2) == names[::2]  # frames 770, 772, 774
    assert cornweed.thin_frames(names, 3) == [names[1], names[4]]  # frames 771, 774
    with pytest.raises(ValueError, match="frame_stride"):
        cornweed.thin_frames(names, 0)


def test_label_map_maize_is_a_crop_and_weeds_are_plants():
    label_map = LabelMap(resolve("configs/label_maps/cornweed.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    assert (label_map.category("Maize"), label_map.category("Weeds")) == ("ZEAMX", "1PLAK")
    assert label_map.rows["Maize"]["role"] == "crop"
