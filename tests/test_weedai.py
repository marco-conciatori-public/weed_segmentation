import json

import pytest

from weedseg.data import weedai
from weedseg.data.coco import BOX_ONLY, LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

TAXA = load_taxa(resolve("configs/eppo_taxa.csv"))
VCD_STEMS = frozenset({"crop: phaseolus vulgaris (stem)", "crop: zea mays (stem)"})


def write_part(raw, part, images, categories):
    """A Weed-AI upload: images as (file_name, width, height, [(category, x, y, w, h)])."""
    coco = {"images": [], "annotations": [], "agcontexts": [], "info": {},
            "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]}
    ids = {name: i for i, name in enumerate(categories)}
    for image_id, (file_name, width, height, boxes) in enumerate(images):
        coco["images"].append({"id": image_id, "file_name": file_name, "width": width,
                               "height": height, "agcontext_id": 0, "license": 0})
        for category, *bbox in boxes:
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": image_id,
                                        "category_id": ids[category], "bbox": bbox,
                                        "segmentation": []})
    (raw / part).mkdir(parents=True)
    (raw / part / "weedcoco.json").write_text(json.dumps(coco), encoding="utf-8")


def write_vcd(tmp_path):
    """Two VCD uploads; row 2020-10-01_lanxade_bean_2 is in both, at two image sizes."""
    raw = tmp_path / "vcd"
    bean, stem = "crop: phaseolus vulgaris (whole-plant)", "crop: phaseolus vulgaris (stem)"
    write_part(raw, "beans", [
        ("2019-05-23_montoldre_bean_1_000000_jpg.rf.0a1b2c.jpg", 2048, 1536,
         [(bean, 10, 10, 200, 150), (stem, 90, 60, 30, 30)]),
        ("2020-10-01_lanxade_bean_2_000004_jpg.rf.99ff.jpg", 1024, 768,
         [(bean, 0, 0, 30, 20)]),  # 30 px at 0.56 mm/px: 16.7 mm
    ], [bean, stem])
    write_part(raw, "maize", [
        ("images/2020-10-01_lanxade_bean_2_000009_jpg.rf.12ab.jpg", 2048, 1536,
         [("crop: zea mays (whole-plant)", 5, 5, 300, 300)]),
    ], ["crop: zea mays (whole-plant)", "crop: zea mays (stem)"])
    return raw


def test_source_names_drop_roboflows_ending():
    assert weedai.source_name("bbro_bbro_07_05_2021_v_0_0_png.rf.2b1646da.jpg") == \
        "bbro_bbro_07_05_2021_v_0_0"
    assert weedai.source_name("images/2019-05-23_montoldre_bean_1_000000_jpg.rf.12.jpg") == \
        "2019-05-23_montoldre_bean_1_000000"
    assert weedai.source_name("UASs_1014.jpg") == "UASs_1014"
    assert weedai.source_name("images/0b2a1f5aa995b471e2a5.jpg") == "0b2a1f5aa995b471e2a5"


def test_groups_from_original_names():
    row = weedai.Grouping(r"^(.+)_\d+$")
    assert row.group_of("2019-05-23_montoldre_bean_1_000003") == "2019-05-23_montoldre_bean_1"
    assert row.group_of("2019-09-25_montoldre_bean_000012") == "2019-09-25_montoldre_bean"
    day = weedai.Grouping(r"^([a-z0-9]+_[a-z0-9]+_\d\d_\d\d_\d{4})_")
    assert day.group_of("walledgarden_walledgarden_06_05_2021_i_12") == \
        day.group_of("walledgarden_walledgarden_06_05_2021_v_1_3") == \
        "walledgarden_walledgarden_06_05_2021"
    blocks = weedai.Grouping(r"^UASs_(\d+)$", block=20)
    assert blocks.group_of("UASs_1014") == blocks.group_of("UASs_1000") == "1000-1019"
    assert blocks.group_of("UASs_1020") == "1020-1039"
    with pytest.raises(ValueError, match="group_pattern"):
        blocks.group_of("DJI_0001")


def test_scale_fixed_or_from_the_footprint():
    assert weedai.Scale(mm_per_px=1.0).of(4864, 3648) == 1.0
    assert weedai.Scale(footprint_mm=570).of(1024, 768) == pytest.approx(0.557, abs=1e-3)
    assert weedai.Scale(footprint_mm=570).of(632, 632) == pytest.approx(0.902, abs=1e-3)
    for both_or_none in ({}, {"mm_per_px": 1.0, "footprint_mm": 570}):
        with pytest.raises(ValueError, match="exactly one"):
            weedai.Scale(**both_or_none)


def test_names_boxes_and_records(tmp_path):
    raw = write_vcd(tmp_path)
    names = weedai.image_names(raw, ["beans", "maize"])
    assert names == ["beans/2019-05-23_montoldre_bean_1_000000_jpg.rf.0a1b2c.jpg",
                     "beans/2020-10-01_lanxade_bean_2_000004_jpg.rf.99ff.jpg",
                     "maize/2020-10-01_lanxade_bean_2_000009_jpg.rf.12ab.jpg"]
    # Stems are not plants.
    assert weedai.box_labels(raw, names[0], VCD_STEMS) == [
        ("crop: phaseolus vulgaris (whole-plant)", 200, 150)]
    assert len(weedai.box_labels(raw, names[0])) == 2
    records = list(weedai.iter_records(raw, names, weedai.Grouping(r"^(.+)_\d+$"),
                                       weedai.Scale(footprint_mm=570), VCD_STEMS))
    first, small, maize = records
    assert (first.file_name, first.width, first.height, first.group) == (
        "beans/images/2019-05-23_montoldre_bean_1_000000_jpg.rf.0a1b2c.jpg", 2048, 1536,
        "2019-05-23_montoldre_bean_1")
    assert first.attributes == {"part": "beans", "source_name": "2019-05-23_montoldre_bean_1_000000",
                                "mm_per_px": 570 / 2048, "mm_per_px_estimated": True}
    assert [(i.source_label, i.bbox_xyxy, i.mask) for i in first.instances] == [
        ("crop: phaseolus vulgaris (whole-plant)", (10, 10, 210, 160), None)]
    # The "images/" folder in a file name is not repeated; a row in two uploads is one group.
    assert maize.file_name == "maize/images/2020-10-01_lanxade_bean_2_000009_jpg.rf.12ab.jpg"
    assert small.group == maize.group == "2020-10-01_lanxade_bean_2"
    with pytest.raises(FileNotFoundError, match="leeks"):
        weedai.image_names(raw, ["leeks"])


def test_byte_identical_images_are_duplicates_of_the_first(tmp_path):
    # ImageWeeds uploaded some photos twice under two numbers.
    raw = tmp_path / "imageweeds"
    weed = "weed: bassia scoparia"
    write_part(raw, "flax", [(f"UASs_{n}.jpg", 40, 30, [(weed, 1, 1, 5, 5)])
                             for n in (1010, 1011, 1323)], [weed])
    (raw / "flax" / "images").mkdir()
    for n, content in [(1010, b"photo a"), (1011, b"photo b"), (1323, b"photo a")]:
        (raw / "flax" / "images" / f"UASs_{n}.jpg").write_bytes(content)
    names = weedai.image_names(raw, ["flax"])
    assert weedai.duplicates(raw, names) == ["flax/UASs_1323.jpg"]


def test_conversion_uses_each_images_own_scale(tmp_path):
    raw = write_vcd(tmp_path)
    grouping, scale = weedai.Grouping(r"^(.+)_\d+$"), weedai.Scale(footprint_mm=570)
    label_map = LabelMap(resolve("configs/label_maps/vcd.csv"), "species", TAXA)
    settings = {"output_dir": str(tmp_path / "out"), "min_plant_size_mm": 20,
                "keep_negative_images": False,
                "split": {"seed": 0, "fractions": {"train": 0.5, "val": 0.5}}}
    names = weedai.image_names(raw, ["beans", "maize"])
    out = convert_dataset(
        names, lambda name: grouping.group_of(weedai.source_name(name)),
        lambda name: weedai.box_labels(raw, name, VCD_STEMS),
        lambda split_names: weedai.iter_records(raw, split_names, grouping, scale, VCD_STEMS),
        label_map, settings, {"name": "VCD"}, box_only=True,
        mm_per_px_of=lambda name: weedai.mm_per_px(raw, name, scale))
    images = [image for split in ("train", "val")
              for image in json.loads((out / f"{split}.json").read_text(encoding="utf-8"))["images"]]
    # The 1024 px image's only bean, 30 px, is 16.7 mm at that image's 0.56 mm/px: under
    # 20 mm, so the image is left without plants and dropped.
    assert sorted(i["source_name"] for i in images) == [
        "2019-05-23_montoldre_bean_1_000000", "2020-10-01_lanxade_bean_2_000009"]
    assert all(image[BOX_ONLY] for image in images)


@pytest.mark.parametrize("path, crop", [("lincolnbeet.csv", "BEAVA"), ("vcd.csv", "ZEAMX"),
                                        ("vcd.csv", "ALLPO")])
def test_crops_keep_their_species(path, crop):
    label_map = LabelMap(resolve(f"configs/label_maps/{path}"), "species", TAXA)
    assert crop in {c["name"] for c in label_map.categories() if c["supercategory"] == "crop"}


def test_lincolnbeets_weeds_are_partial_labels():
    label_map = LabelMap(resolve("configs/label_maps/lincolnbeet.csv"), "species", TAXA)
    assert label_map.category("weed: UNSPECIFIED") == "1PLAK"
    assert label_map.limits("weed: UNSPECIFIED") == {"none_of": ["BEAVA"]}
