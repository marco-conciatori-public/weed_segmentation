import cv2
import numpy as np
import pytest

from weedseg.data import sugarbeets2016 as sb
from weedseg.data.coco import LabelMap
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CLASSES = (("beet", (10000,)), ("weed", (2, 20100)))
NAME = "CKA_160523/images/rgb/bonirob_2016-05-23-10-47-22_2_frame200.png"


def write_image(raw, name, labels):
    root = raw / sb.ROOT
    image = root / name
    image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), np.zeros((*labels.shape, 3), np.uint8))
    mask = sb.label_path(raw, name)
    mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask), labels.astype(np.uint16))


def test_names_and_runs(tmp_path):
    labels = np.zeros((40, 60), np.uint16)
    write_image(tmp_path, NAME, labels)
    # an image without its index map is left out
    lonely = tmp_path / sb.ROOT / NAME.replace("frame200", "frame201")
    cv2.imwrite(str(lonely), np.zeros((40, 60, 3), np.uint8))
    assert sb.image_names(tmp_path) == [NAME]
    assert sb.parse(NAME) == {"run": "bonirob_2016-05-23-10-47-22_2", "date": "2016-05-23",
                              "frame": 200}
    assert sb.group_of(NAME) == "bonirob_2016-05-23-10-47-22_2"
    with pytest.raises(ValueError, match="Unexpected"):
        sb.parse("CKA_160523/images/rgb/other.png")


def test_patches_become_boxes(tmp_path):
    labels = np.zeros((40, 60), np.uint16)
    labels[5:15, 10:30] = 10000  # one beet
    labels[14:20, 28:32] = 10000  # touching it: the same patch
    labels[30:35, 5:10] = 2  # a weed
    labels[30:35, 40:45] = 20100  # another weed value
    labels[0, 59] = 2  # a speck
    write_image(tmp_path, NAME, labels)
    size, found = sb.patches(tmp_path, NAME, CLASSES, 5)
    assert size == (60, 40)
    assert found == (("beet", (10, 5, 32, 20), 222), ("weed", (5, 30, 10, 35), 25),
                     ("weed", (40, 30, 45, 35), 25))
    assert sb.box_labels(tmp_path, NAME, CLASSES, 5)[0] == ("beet", 22, 15)
    (record,) = sb.iter_records(tmp_path, [NAME], CLASSES, 5, 0.33)
    assert (record.file_name, record.width, record.height) == (NAME, 60, 40)
    assert record.attributes == {"run": "bonirob_2016-05-23-10-47-22_2", "date": "2016-05-23",
                                 "frame": 200, "mm_per_px": 0.33, "mm_per_px_estimated": True}
    assert [(i.source_label, i.attributes["patch_px"]) for i in record.instances] == [
        ("beet", 222), ("weed", 25), ("weed", 25)]


def test_label_map_covers_the_classes():
    label_map = LabelMap(resolve("configs/label_maps/sugarbeets2016.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    assert (label_map.category("beet"), label_map.category("weed")) == ("BEAVA", "1PLAK")
