import pytest

from weedseg.data.coco import IGNORE_CATEGORY, OTHER_CATEGORY
from weedseg.data.combine import combine


def taxon(code, rank, genus, family="1GRAF"):
    return {"eppo": code, "rank": rank, "name": f"name of {code}", "genus": genus,
            "family": family, "order": "1POAO"}


TAXA = {t["eppo"]: t for t in [
    taxon("ZEAMX", "species", "1ZEAG"), taxon("POAAN", "species", "1POAG"),
    taxon("POATR", "species", "1POAG"), taxon("1POAG", "genus", "1POAG"),
    taxon("1MERG", "genus", "1MERG", "1EUPF"), taxon("SETVI", "species", "1SETG")]}


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
    assert [i["file_name"] for i in train["images"]] == [
        "data/raw/a/images/1.jpg", "data/raw/a/images/2.jpg",
        "data/raw/b/images/1.jpg", "data/raw/b/images/2.jpg"]
    assert len({i["id"] for i in train["images"]}) == 4
    assert train["info"]["image_dir"] == "."
    assert report[0].startswith("2 classes")


def test_size_filtered_plants_keep_their_code_and_stay_ignored():
    a = dataset([("ZEAMX", "crop")], [["ZEAMX"]])
    ignore_id = a["train"]["categories"][-1]["id"]
    a["train"]["annotations"].append({"id": 9, "image_id": 1, "iscrowd": 1, "bbox": [0, 0, 1, 1],
                                      "category_id": ignore_id, "ignored_category_id": 1})
    combined, _ = combine({"a": a}, TAXA, min_images=1)
    small = combined["train"]["annotations"][-1]
    assert small["eppo"] == "ZEAMX" and small["iscrowd"] == 1
    assert "ignored_category_id" not in small
    assert combined["train"]["categories"][small["category_id"] - 1]["name"] == IGNORE_CATEGORY


def test_combine_refuses_mixed_sizes_and_non_species_categories():
    with pytest.raises(ValueError, match="min_plant_size_mm"):
        combine({"a": dataset([("ZEAMX", "crop")], [["ZEAMX"]]),
                 "b": dataset([("ZEAMX", "crop")], [["ZEAMX"]], size=30)}, TAXA, 1)
    with pytest.raises(ValueError, match="not an EPPO code"):
        combine({"a": dataset([("grasses", "weed")], [["grasses"]])}, TAXA, 1)
