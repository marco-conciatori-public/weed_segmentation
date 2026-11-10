import pytest

from weedseg.data import weedmaize
from weedseg.data.coco import LabelMap
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

# Plants in photo coordinates: (label, x0, y0, x1, y1), 1-based inclusive like VOC.
PLANTS = [("ZEAMX_V1", 229, 814, 441, 1017), ("SOLNI_V1", 600, 300, 640, 350),
          ("CYPRO", 1500, 200, 1560, 260), ("ZEAMX_V1", 2453, 974, 2507, 1043),
          ("ECHCG", 3100, 700, 3150, 760), ("NC", 3900, 400, 3940, 440)]
SIZE = (3000, 1296)


def tile_xml(origin, extra=()):
    """The VOC file of the tile at `origin` in the photo: the plants inside it, clipped and
    flagged truncated at its edges."""
    ox, oy = origin
    objects = []
    for label, x0, y0, x1, y1 in [*PLANTS, *extra]:
        cx0, cy0 = max(x0 - ox, 1), max(y0 - oy, 1)
        cx1, cy1 = min(x1 - ox, SIZE[0]), min(y1 - oy, SIZE[1])
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        truncated = int((cx0, cy0, cx1, cy1) != (x0 - ox, y0 - oy, x1 - ox, y1 - oy))
        objects.append(f"<object><name>{label}</name><truncated>{truncated}</truncated>"
                       f"<bndbox><xmin>{cx0}</xmin><ymin>{cy0}</ymin><xmax>{cx1}</xmax>"
                       f"<ymax>{cy1}</ymax></bndbox></object>")
    return (f"<annotation><size><width>{SIZE[0]}</width><height>{SIZE[1]}</height></size>"
            + "".join(objects) + "</annotation>")


def write_photo(raw, folder, photo, origins, extra=()):
    (raw / folder).mkdir(parents=True, exist_ok=True)
    for tile, origin in origins.items():
        (raw / folder / f"IMG_{photo}_{tile:02d}.xml").write_text(tile_xml(origin, extra),
                                                                  encoding="utf-8")
        (raw / folder / f"IMG_{photo}_{tile:02d}.JPG").write_bytes(b"")


ORIGINS = {1: (0, 0), 2: (400, 0), 3: (800, 0), 4: (1200, 0), 5: (1600, 0), 6: (2000, 0)}


def test_names_and_reading(tmp_path):
    write_photo(tmp_path, "validation_set", 1822, {1: (0, 0)})
    write_photo(tmp_path, "train_set", 1900, {1: (0, 0)})
    (tmp_path / "test_set").mkdir()
    names = weedmaize.image_names(tmp_path)
    assert names == ["train_set/IMG_1900_01", "validation_set/IMG_1822_01"]
    assert weedmaize.parse(names[1]) == ("validation_set", 1822, 1)
    size, boxes = weedmaize.read_tile(tmp_path, names[1])
    assert size == SIZE
    assert boxes[0] == ("ZEAMX_V1", (228, 813, 441, 1017), False)  # 0-based, end exclusive
    assert weedmaize.group_of(names[1], 10) == "IMG_1820-1829"


def test_duplicate_boxes_are_dropped():
    boxes = (("ZEAMX_V1", (228, 813, 441, 1017), False),
             ("ZEAMX_V1", (226, 811, 439, 1015), False),  # the same plant boxed again
             ("SOLNI", (230, 820, 260, 850), False))  # a weed under the maize leaves
    kept = weedmaize.drop_duplicates(boxes, 0.9)
    assert [b[1] for b in kept] == [(228, 813, 441, 1017), (230, 820, 260, 850)]


def test_tile_grid_and_spread():
    assert [weedmaize.tile_offset(t) for t in (1, 2, 6, 7, 8, 42)] == [
        (0, 0), (400, 0), (2000, 0), (0, 400), (400, 400), (2000, 2400)]
    # 3000x1296 tiles: columns 0 and 2000 share a third, rows 0, 1200 and 2400 share 7%
    assert weedmaize.spread_tiles(list(range(1, 43)), SIZE, 0.35) == [1, 6, 19, 24, 37, 42]
    assert weedmaize.spread_tiles(list(range(1, 43)), SIZE, 0.25) == [1, 12, 19, 30, 37]
    assert weedmaize.spread_tiles([2, 3, 6], SIZE, 0.35) == [2]  # 2 and 6 share 47%
    assert weedmaize.spread_tiles([2, 3, 6], SIZE, 0.5) == [2, 6]
    # a missing tile is replaced by the one that covers most of what is left
    assert weedmaize.spread_tiles([2, 6, 13, 19, 24, 37, 42], SIZE, 0.35) == [
        2, 19, 24, 37, 42]


def test_tile_offsets_match_the_boxes_the_tiles_share(tmp_path):
    write_photo(tmp_path, "train_set", 1822, ORIGINS)
    for tile, origin in ORIGINS.items():
        assert weedmaize.tile_offset(tile) == origin
        boxes = weedmaize.read_tile(tmp_path, f"train_set/IMG_1822_{tile:02d}")[1]
        whole = [b for b in boxes if not b[2]]
        for label, (x0, y0, x1, y1), _ in whole:  # back in photo coordinates
            assert (label, x0 + origin[0] + 1, y0 + origin[1] + 1, x1 + origin[0],
                    y1 + origin[1]) in PLANTS


def test_select_tiles_across_sets(tmp_path):
    # the authors' sets mix tiles of one photo; the choice is made over all of them
    write_photo(tmp_path, "train_set", 1822, {1: (0, 0), 2: (400, 0)})
    write_photo(tmp_path, "test_set", 1822, {6: (2000, 0)})
    write_photo(tmp_path, "validation_set", 1830, {3: (800, 0)})
    names = weedmaize.image_names(tmp_path)
    assert weedmaize.select_tiles(tmp_path, names, 0.35) == [
        "test_set/IMG_1822_06", "train_set/IMG_1822_01", "validation_set/IMG_1830_03"]


def test_records(tmp_path):
    write_photo(tmp_path, "test_set", 2081, {1: (0, 0)},
                extra=[("ZEAMX_V1", 228, 812, 440, 1016),  # a duplicate of the first maize
                       ("ECHCG", 2950, 100, 3050, 200)])  # cut by the tile's edge
    (record,) = weedmaize.iter_records(tmp_path, ["test_set/IMG_2081_01"], 10,
                                       weedmaize.Labelling(0.9), 0.36)
    assert (record.file_name, record.width, record.height) == ("test_set/IMG_2081_01.JPG",
                                                               *SIZE)
    assert record.group == "IMG_2080-2089"
    assert record.attributes == {"photo": 2081, "tile": 1, "set": "test_set",
                                 "mm_per_px": 0.36, "mm_per_px_estimated": True}
    labels = [i.source_label for i in record.instances]
    # the far ECHCG and NC lie outside the tile, the duplicate maize is dropped
    assert labels == ["ZEAMX_V1", "SOLNI_V1", "CYPRO", "ZEAMX_V1", "ECHCG"]
    assert record.instances[-1].attributes == {"truncated": True}
    assert record.instances[-1].bbox_xyxy == (2949, 99, 3000, 200)


def test_centre_boxes_of_a_plant_are_dropped_or_ignored(tmp_path):
    write_photo(tmp_path, "train_set", 1900, {1: (0, 0)}, extra=[
        ("CYPRO_max", 1000, 500, 1300, 800), ("CYPRO_min", 1120, 620, 1180, 680),  # one plant
        ("SOLNI_V1", 1010, 510, 1040, 540),  # a weed under its leaves: a plant of its own
        ("CYPRO_min", 2000, 900, 2040, 940)])  # a centre without its whole plant
    labelling = weedmaize.Labelling(0.9, {"CYPRO_min": "CYPRO_max"}, 0.5)
    name = "train_set/IMG_1900_01"
    (record,) = weedmaize.iter_records(tmp_path, [name], 10, labelling)
    extra = [(i.source_label, i.ignore, i.attributes) for i in record.instances[4:]]
    assert extra == [("CYPRO_max", False, {}), ("SOLNI_V1", False, {}),
                     ("CYPRO_min", True, {"centre_only": True})]
    labels = [label for label, _, _ in weedmaize.box_labels(tmp_path, name, labelling)]
    assert labels[4:] == ["CYPRO_max", "SOLNI_V1"]  # the ignore region is no plant to count
    # without the setting both boxes stay
    plain = weedmaize.box_labels(tmp_path, name, weedmaize.Labelling(0.9))
    assert [label for label, _, _ in plain][4:] == ["CYPRO_max", "CYPRO_min", "SOLNI_V1",
                                                    "CYPRO_min"]


def test_label_map_covers_the_classes():
    label_map = LabelMap(resolve("configs/label_maps/weedmaize.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    assert {label_map.category(c) for c in ("ZEAMX", "ZEAMX_V1", "ZEAMX_V4")} == {"ZEAMX"}
    assert label_map.category("NC") == label_map.category("OE") == "1PLAK"
    assert label_map.category("ECHCG_Ve") == "ECHCG"
    with pytest.raises(KeyError):
        label_map.category("XYZ")
