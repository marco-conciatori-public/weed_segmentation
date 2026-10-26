import json

from weedseg.data import groundbased_weed
from weedseg.data.coco import BOX_ONLY, LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

# A tomato, a nightshade (truncated by the border) and a plant not recognised.
OBJECTS = [("LYPES", (2.0, 3.0, 10.0, 8.0), "0", "0"), ("SOLNI", (0.0, 12.0, 3.5, 16.0), "0", "1"),
           ("NR", (20.0, 1.0, 22.0, 3.0), "1", "0")]


def write_image(raw, name, objects=OBJECTS):
    part, stem = name.split("/")
    xml_objects = "".join(
        f"<object><name>{label}</name><truncated>{truncated}</truncated><occluded>0</occluded>"
        f"<difficult>{difficult}</difficult><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for label, (x0, y0, x1, y1), difficult, truncated in objects)
    xml = (f"<annotation><folder>IMG</folder><filename>{stem}.JPG</filename>"
           f"<size><width>30</width><height>20</height><depth></depth></size>"
           f"<segmented>0</segmented>{xml_objects}</annotation>")
    (raw / part / "Annotations").mkdir(parents=True, exist_ok=True)
    (raw / part / "Annotations" / f"{stem}.xml").write_text(xml, encoding="utf-8")


def write_dataset(tmp_path):
    raw = tmp_path / "groundbased_weed"
    write_image(raw, "TOMATO_1/IMG_4130")
    write_image(raw, "TOMATO_1/IMG_4139")
    write_image(raw, "TOMATO_1/02_IMG_4550")
    write_image(raw, "TOMATO_1/25IMG_4626")
    write_image(raw, "TOMATO_2/IMG_5140", objects=[("ECHCG", (1, 1, 9, 9), "0", "0")])
    return raw


def test_names_groups_and_boxes(tmp_path):
    raw = write_dataset(tmp_path)
    names = groundbased_weed.image_names(raw)
    assert names == ["TOMATO_1/02_IMG_4550", "TOMATO_1/25IMG_4626", "TOMATO_1/IMG_4130",
                     "TOMATO_1/IMG_4139", "TOMATO_2/IMG_5140"]
    # Blocks of consecutive camera numbers, quadrat cut-outs by their photo's number.
    assert [groundbased_weed.group_of(name, 10) for name in names] == [
        "TOMATO_1_4550", "TOMATO_1_4620", "TOMATO_1_4130", "TOMATO_1_4130", "TOMATO_2_5140"]
    assert groundbased_weed.box_labels(raw, "TOMATO_1/IMG_4130") == [
        ("LYPES", 8, 5), ("SOLNI", 3.5, 4), ("NR", 2, 2)]


def test_records(tmp_path):
    raw = write_dataset(tmp_path)
    full, quadrat = groundbased_weed.iter_records(raw, ["TOMATO_1/IMG_4130",
                                                        "TOMATO_1/02_IMG_4550"], 10, 0.3)
    assert (full.file_name, full.width, full.height, full.group) == (
        "TOMATO_1/Images/IMG_4130.JPG", 30, 20, "TOMATO_1_4130")
    assert full.attributes == {"subset": "TOMATO_1", "year": 2021, "mm_per_px": 0.3,
                               "mm_per_px_estimated": True}
    assert quadrat.attributes["quadrat"] == 2
    tomato, nightshade, unknown = full.instances
    assert tomato.bbox_xyxy == (2, 3, 10, 8) and tomato.mask is None and not tomato.attributes
    assert nightshade.attributes == {"truncated": True}
    assert unknown.attributes == {"difficult": True} and not unknown.ignore


def test_conversion_marks_box_only_and_maps_unrecognised_plants(tmp_path):
    raw = write_dataset(tmp_path)
    names = groundbased_weed.image_names(raw)
    label_map = LabelMap(resolve("configs/label_maps/groundbased_weed.csv"), "species",
                         load_taxa(resolve("configs/eppo_taxa.csv")))
    assert label_map.category("NR") == "1PLAK"  # plants not recognised: species unknown
    assert label_map.rows["LYPES"]["role"] == "crop"
    settings = {"output_dir": str(tmp_path / "out"), "min_plant_size_mm": 1.5,
                "mm_per_px_estimate": 0.5, "keep_negative_images": False,
                "split": {"seed": 0, "fractions": {"train": 0.5, "val": 0.5}}}
    out = convert_dataset(names, lambda name: groundbased_weed.group_of(name, 10),
                          lambda name: groundbased_weed.box_labels(raw, name),
                          lambda split_names: groundbased_weed.iter_records(raw, split_names,
                                                                            10, 0.5),
                          label_map, settings, {"name": "GROUNDBASED_WEED"}, box_only=True)
    files = [json.loads((out / f"{split}.json").read_text(encoding="utf-8"))
             for split in ("train", "val")]
    images = [image for coco in files for image in coco["images"]]
    assert len(images) == 5 and all(image[BOX_ONLY] for image in images)
    coco = next(coco for coco in files if any(i["file_name"].endswith("IMG_4130.JPG")
                                              for i in coco["images"]))
    # The two photos of one block stay together.
    assert any(i["file_name"].endswith("IMG_4139.JPG") for i in coco["images"])
    image_id = next(i["id"] for i in coco["images"] if i["file_name"].endswith("IMG_4130.JPG"))
    names_of = {c["id"]: c["name"] for c in coco["categories"]}
    # NR (2 px, 1 mm at 0.5 mm/px) is under 1.5 mm: an ignore region.
    assert [(names_of[a["category_id"]], a["iscrowd"], a.get("truncated", False))
            for a in coco["annotations"] if a["image_id"] == image_id] == [
        ("LYPES", 0, False), ("SOLNI", 0, True), ("ignore", 1, False)]
    assert "segmentation" not in coco["annotations"][0]
