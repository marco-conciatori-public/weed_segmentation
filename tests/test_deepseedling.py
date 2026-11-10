import json

import pytest
from PIL import Image

from weedseg.data import deepseedling
from weedseg.data.coco import LabelMap
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

SETS = {"TAMU2015": {"mm_per_px": 0.4, "frame_stride": 2, "group_block": 2},
        "UGA2015": {"mm_per_px": 0.35, "frame_stride": 2, "group_block": 2},
        "UGA2018": {"mm_per_px": 0.35, "frame_stride": 1, "group_block": 2}}


def box(x1, y1, x2, y2, tag, width=70, height=54):
    """A VoTT box drawn on a 70x54 screen (images here are 140x108: twice as large)."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "id": 0, "width": width,
            "height": height, "type": "Rectangle", "tags": [tag], "name": 1}


def write_folder(root, set_name, folder, files, frames):
    images = root / set_name / folder
    images.mkdir(parents=True)
    for name in files:
        Image.new("RGB", (140, 108)).save(images / name)
    (root / set_name / f"{folder}.json").write_text(
        json.dumps({"frames": frames, "inputTags": "Plant,Weed"}), encoding="utf-8")


def write_dataset(tmp_path):
    # VoTT numbered the frames as Windows lists them: 1001, 121, 131 (by name, not number)
    write_folder(tmp_path, "UGA2015", "102", ["121.jpg", "131.jpg", "1001.jpg"], {
        "0": [box(10, 5, 20, 15, "Plant")],              # 1001.jpg
        "1": [],                                         # 121.jpg: no boxes, not used
        "2": [box(30, 20, 35, 30, "Weed"), box(40, 30, 60, 40, "Plant")]})  # 131.jpg
    # one folder, two videos (file-name prefixes); frame 2 was never labelled
    write_folder(tmp_path, "UGA2018", "Images",
                 ["ENGR_1_000081.jpg", "ENGR_1_000091.jpg", "NAM3_2_000011.jpg"], {
                     "0": [box(1, 1, 5, 5, "Plant")], "1": [box(2, 2, 6, 6, "Plant")]})
    write_folder(tmp_path, "TAMU2015", "Row20", ["Frame00001.JPG", "Frame00007.JPG"], {
        "0": [box(1, 1, 9, 9, "Weed")], "1": [box(1, 1, 9, 9, "Plant")]})
    # labelled on a video file, no folder of frames: ignored
    (tmp_path / "TAMU2015" / "Row10.mp4.json").write_text(
        json.dumps({"frames": {"0": [box(1, 1, 9, 9, "Plant")]}}), encoding="utf-8")
    return tmp_path


def test_parse_video_and_frame():
    assert deepseedling.parse("UGA2018/Images/ENGR_1_000081.jpg") == {
        "set": "UGA2018", "folder": "Images", "video": "UGA2018/Images/ENGR_1", "frame": 81}
    assert deepseedling.parse("UGA2015/102/1001.jpg")["video"] == "UGA2015/102"
    assert deepseedling.parse("TAMU2015/Row20/Frame05461.JPG")["frame"] == 5461
    with pytest.raises(ValueError, match="Unexpected"):
        deepseedling.parse("1001.jpg")


def test_frames_map_to_files_in_vott_order_and_boxes_scale(tmp_path):
    raw = write_dataset(tmp_path)
    names = deepseedling.image_names(raw, SETS)
    assert names == ["TAMU2015/Row20/Frame00001.JPG", "TAMU2015/Row20/Frame00007.JPG",
                     "UGA2015/102/1001.jpg", "UGA2015/102/131.jpg",
                     "UGA2018/Images/ENGR_1_000081.jpg", "UGA2018/Images/ENGR_1_000091.jpg"]
    # boxes drawn on the 70x54 screen, in px of the 140x108 image
    assert deepseedling.box_labels(raw, "UGA2015/102/131.jpg", SETS) == [
        ("Weed", 10, 20), ("Plant", 40, 20)]
    record = next(deepseedling.iter_records(raw, ["UGA2015/102/1001.jpg"], SETS))
    assert [(i.source_label, i.bbox_xyxy) for i in record.instances] == [
        ("Plant", (20, 10, 40, 30))]
    assert (record.width, record.height) == (140, 108)
    assert record.attributes["mm_per_px"] == 0.35 and record.attributes["frame"] == 1001


def test_boxes_are_clipped_and_bad_tags_refused(tmp_path):
    write_folder(tmp_path, "UGA2015", "1", ["1.jpg"],
                 {"0": [box(60, 50, 90, 70, "Plant"), box(80, 1, 90, 5, "Plant")]})
    double = {**box(1, 1, 5, 5, "Weed"), "tags": ["Weed", "Plant"]}
    write_folder(tmp_path, "UGA2015", "3", ["1.jpg"], {"0": [double]})
    sets = {"UGA2015": SETS["UGA2015"]}
    # the first box is clipped to the image, the second lies outside it and is dropped
    assert deepseedling.box_labels(tmp_path, "UGA2015/1/1.jpg", sets) == [("Plant", 20, 8)]
    # a box tagged both ways is a plant of unknown species
    assert deepseedling.box_labels(tmp_path, "UGA2015/3/1.jpg", sets) == [("Plant+Weed", 8, 8)]
    write_folder(tmp_path, "UGA2018", "2", ["1.jpg"], {"0": [box(1, 1, 5, 5, "Grass")]})
    with pytest.raises(ValueError, match="tags"):
        deepseedling.image_names(tmp_path, {"UGA2018": SETS["UGA2018"]})


def test_thinning_and_groups_follow_time_order_per_video(tmp_path):
    raw = write_dataset(tmp_path)
    names = deepseedling.image_names(raw, SETS)
    # UGA2015 video 102 in time order: 131 (position 0), 1001 (position 1); stride 2 keeps
    # 131. UGA2018: stride 1 keeps both; TAMU2015: stride 2 keeps the first frame.
    assert deepseedling.thin_frames(raw, names, SETS) == [
        "TAMU2015/Row20/Frame00001.JPG", "UGA2015/102/131.jpg",
        "UGA2018/Images/ENGR_1_000081.jpg", "UGA2018/Images/ENGR_1_000091.jpg"]
    assert (deepseedling.group_of(raw, "UGA2015/102/131.jpg", SETS)
            == deepseedling.group_of(raw, "UGA2015/102/1001.jpg", SETS) == "UGA2015/102_0000")
    assert deepseedling.group_of(raw, "UGA2018/Images/ENGR_1_000091.jpg", SETS) == (
        "UGA2018/Images/ENGR_1_0000")
    with pytest.raises(ValueError, match="frame_stride"):
        deepseedling.thin_frames(raw, names, {**SETS, "UGA2018": {**SETS["UGA2018"],
                                                                  "frame_stride": 0}})


def test_frame_beyond_the_folder_is_an_error(tmp_path):
    write_folder(tmp_path, "UGA2015", "7", ["1.jpg"], {"3": [box(1, 1, 5, 5, "Plant")]})
    with pytest.raises(ValueError, match="frame 3"):
        deepseedling.image_names(tmp_path, {"UGA2015": SETS["UGA2015"]})


def test_missing_set_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_datasets"):
        deepseedling.image_names(tmp_path, {"UGA2015": SETS["UGA2015"]})


def test_label_map_cotton_is_a_crop_and_weeds_are_plants():
    label_map = LabelMap(resolve("configs/label_maps/deepseedling.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    assert (label_map.category("Plant"), label_map.category("Weed"),
            label_map.category("Plant+Weed")) == ("GOSHI", "1PLAK", "1PLAK")
    assert label_map.rows["Plant"]["role"] == "crop"
