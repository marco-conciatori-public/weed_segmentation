import pytest

from weedseg.data.coco import BOX_ONLY, IGNORE_CATEGORY, OTHER_CATEGORY, drop_box_only
from weedseg.data.combine import allowed_classes, combine, describe_classes
from weedseg.data.plant import to_plant_only


def taxon(code, rank, genus, family="1GRAF"):
    return {"eppo": code, "rank": rank, "name": f"name of {code}", "genus": genus,
            "family": family, "order": "1POAO"}


TAXA = {t["eppo"]: t for t in [
    taxon("ZEAMX", "species", "1ZEAG"), taxon("POAAN", "species", "1POAG"),
    taxon("POATR", "species", "1POAG"), taxon("1POAG", "genus", "1POAG"),
    taxon("1MERG", "genus", "1MERG", "1EUPF"), taxon("SETVI", "species", "1SETG"),
    taxon("1SETG", "genus", "1SETG"),
    {"eppo": "1PLAK", "rank": "kingdom", "name": "Plantae", "genus": "", "family": "",
     "order": ""}]}


def dataset(categories, train_images, image_dir="data/raw/a/images", size=20):
    """A converted dataset: `categories` is [(code, role)], `train_images` a list of the codes
    of each training image's plants. val and test hold one image without plants."""
    cats = [{"id": i + 1, "name": code, "supercategory": role}
            for i, (code, role) in enumerate(categories)]
    cats.append({"id": len(cats) + 1, "name": IGNORE_CATEGORY, "supercategory": "ignore"})
    ids = {c["name"]: c["id"] for c in cats}
    info = {"image_dir": image_dir, "min_plant_size_mm": size, "licence": "CC BY 4.0"}
    images, annotations = [], []
    for i, codes in enumerate(train_images, start=1):
        images.append({"id": i, "file_name": f"{i}.jpg"})
        for code in codes:
            annotations.append({"id": len(annotations) + 1, "image_id": i, "iscrowd": 0,
                                "category_id": ids[code], "bbox": [0, 0, 10, 10]})
    empty = {"info": info, "images": [{"id": 1, "file_name": "x.jpg"}], "annotations": [],
             "categories": cats}
    return {"train": {"info": info, "images": images, "annotations": annotations,
                      "categories": cats}, "val": empty, "test": empty}


def test_the_four_rules_on_the_combined_training_images():
    a = dataset([("ZEAMX", "crop"), ("POAAN", "weed"), ("1POAG", "weed")],
                [["ZEAMX", "POAAN"], ["ZEAMX", "1POAG"]])
    b = dataset([("POAAN", "weed"), ("POATR", "weed"), ("1MERG", "weed")],
                [["POAAN", "POATR"], ["1MERG"]], image_dir="data/raw/b/images")
    combined, report = combine({"a": a, "b": b}, TAXA, min_images=2)
    train = combined["train"]
    names = {c["id"]: c["name"] for c in train["categories"]}
    assert list(names.values()) == ["ZEAMX", "POAAN", OTHER_CATEGORY, IGNORE_CATEGORY]
    # POAAN reaches 2 images only over both datasets; POATR (1 image) goes to other_plant;
    # the genus Poa has a class, so its genus-only label becomes ignore; Mercurialis doesn't.
    got = [(a["eppo"], names[a["category_id"]], a["iscrowd"]) for a in train["annotations"]]
    assert got == [("ZEAMX", "ZEAMX", 0), ("POAAN", "POAAN", 0), ("ZEAMX", "ZEAMX", 0),
                   ("1POAG", IGNORE_CATEGORY, 1), ("POAAN", "POAAN", 0),
                   ("POATR", OTHER_CATEGORY, 0), ("1MERG", OTHER_CATEGORY, 0)]
    assert train["annotations"][3].get("hidden_plant") is True  # the genus Poa
    assert [i["file_name"] for i in train["images"]] == [
        "data/raw/a/images/1.jpg", "data/raw/a/images/2.jpg",
        "data/raw/b/images/1.jpg", "data/raw/b/images/2.jpg"]
    assert len({i["id"] for i in train["images"]}) == 4
    assert train["info"]["image_dir"] == "."
    assert report[0].startswith("2 classes")


def test_plants_of_unknown_species_are_ignore_regions_but_plants_for_the_detector():
    a = dataset([("ZEAMX", "crop"), ("1PLAK", "weed")], [["ZEAMX", "1PLAK", "1PLAK"]])
    a["train"]["annotations"][2]["iscrowd"] = 1  # an unknown plant the converter left out
    combined, report = combine({"a": a}, TAXA, min_images=1)
    train = combined["train"]
    names = {c["id"]: c["name"] for c in train["categories"]}
    assert list(names.values()) == ["ZEAMX", IGNORE_CATEGORY]
    got = [(a["eppo"], names[a["category_id"]], a["iscrowd"], a.get("hidden_plant"))
           for a in train["annotations"]]
    assert got == [("ZEAMX", "ZEAMX", 0, None), ("1PLAK", IGNORE_CATEGORY, 1, True),
                   ("1PLAK", IGNORE_CATEGORY, 1, None)]
    assert train["info"]["hidden_plants"] == 1
    assert any(line.lstrip().startswith("1PLAK") and "ignore" in line for line in report)
    plants = to_plant_only(train)
    assert [(a["category_id"], a["iscrowd"]) for a in plants["annotations"]] == [
        (1, 0), (1, 0), (2, 1)]


def test_datasets_with_boxes_only_count_for_the_classes_and_keep_their_mark():
    masks = dataset([("ZEAMX", "crop"), ("POAAN", "weed")], [["ZEAMX", "POAAN"], ["ZEAMX"]])
    for annotation in masks["train"]["annotations"]:
        annotation["segmentation"] = {"size": [10, 10], "counts": "fake"}
    boxes = dataset([("POAAN", "weed")], [["POAAN"]], image_dir="data/raw/b/images")
    for image in boxes["train"]["images"]:
        image[BOX_ONLY] = True
    combined, _ = combine({"a": masks, "b": boxes}, TAXA, min_images=2)
    train = combined["train"]
    names = [c["name"] for c in train["categories"]]
    assert names == ["ZEAMX", "POAAN", IGNORE_CATEGORY]  # POAAN has 2 images only with b
    assert [i.get(BOX_ONLY) for i in train["images"]] == [None, None, True]
    for coco in (train, to_plant_only(train)):  # the detector's copy keeps the mark
        assert [i["file_name"] for i in drop_box_only(coco)["images"]] == [
            "data/raw/a/images/1.jpg", "data/raw/a/images/2.jpg"]


def test_size_filtered_plants_keep_their_code_and_stay_ignored():
    a = dataset([("ZEAMX", "crop")], [["ZEAMX"]])
    ignore_id = a["train"]["categories"][-1]["id"]
    a["train"]["annotations"].append({"id": 9, "image_id": 1, "iscrowd": 1, "bbox": [0, 0, 1, 1],
                                      "category_id": ignore_id, "ignored_category_id": 1})
    combined, _ = combine({"a": a}, TAXA, min_images=1)
    small = combined["train"]["annotations"][-1]
    assert small["eppo"] == "ZEAMX" and small["iscrowd"] == 1
    assert "ignored_category_id" not in small and "hidden_plant" not in small
    assert combined["train"]["categories"][small["category_id"] - 1]["name"] == IGNORE_CATEGORY


def test_combine_refuses_mixed_sizes_and_non_species_categories():
    with pytest.raises(ValueError, match="min_plant_size_mm"):
        combine({"a": dataset([("ZEAMX", "crop")], [["ZEAMX"]]),
                 "b": dataset([("ZEAMX", "crop")], [["ZEAMX"]], size=30)}, TAXA, 1)
    with pytest.raises(ValueError, match="not an EPPO code"):
        combine({"a": dataset([("grasses", "weed")], [["grasses"]])}, TAXA, 1)


def test_allowed_classes_of_hidden_plants():
    classes = ["ZEAMX", "POAAN", "SETVI", OTHER_CATEGORY]
    roles = {"ZEAMX": "crop", "POAAN": "weed", "SETVI": "weed", OTHER_CATEGORY: "other"}

    def allowed(code, role, one_of=(), none_of=()):
        return allowed_classes(code, role, list(one_of), list(none_of), classes, roles, TAXA)

    assert allowed("1POAG", "weed") == ["POAAN", OTHER_CATEGORY]  # Poa: Poa annua or another
    assert allowed("1SETG", "crop") == [OTHER_CATEGORY]  # a crop Setaria isn't the weed
    assert allowed("1PLAK", "weed", none_of=["ZEAMX"]) == ["POAAN", "SETVI", OTHER_CATEGORY]
    assert allowed("1PLAK", "weed", one_of=["1POAG", "1SETG"]) == [
        "POAAN", "SETVI", OTHER_CATEGORY]
    assert allowed("1PLAK", "crop", one_of=["ZEAMX"]) == ["ZEAMX"]  # all its species are classes
    assert allowed("1PLAK", "crop", one_of=["ZEAMX", "PHSVX"]) == ["ZEAMX", OTHER_CATEGORY]
    assert allowed("1PLAK", "unknown") is None  # every class: nothing to learn
    assert allowed("1PLAK", "weed", one_of=["1POAO"]) is None
    assert describe_classes(["POAAN", OTHER_CATEGORY], classes) == "POAAN, other_plant"
    assert describe_classes(["POAAN", "SETVI", OTHER_CATEGORY], classes) == (
        "every class but ZEAMX")


def test_combining_gives_hidden_plants_their_allowed_classes():
    a = dataset([("ZEAMX", "crop"), ("POAAN", "weed"), ("1PLAK", "weed"), ("1POAG", "weed")],
                [["ZEAMX", "POAAN", "1PLAK", "1PLAK", "1POAG"]])
    weed, unknown = a["train"]["annotations"][2:4]
    weed["none_of"] = ["ZEAMX"]  # e.g. a weed of a maize field: anything but maize
    combined, report = combine({"a": a}, TAXA, min_images=1)
    train = combined["train"]
    got = [a.get("allowed_classes") for a in train["annotations"]]
    # Without other_plant, a Poa can only be Poa annua.
    assert got == [None, None, ["POAAN"], None, ["POAAN"]]
    assert unknown.get("none_of") is None and train["annotations"][3]["hidden_plant"]
    assert train["info"]["partial_labels"] == 2
    assert report[-3].startswith("Partial labels")
    assert any("1PLAK" in line and "-> POAAN" in line for line in report[-2:])
    plants = to_plant_only(train)  # the detector learns them as plants all the same
    assert [a["category_id"] for a in plants["annotations"]] == [1] * 5
