import pytest

from weedseg.data.coco import IGNORE_CATEGORY
from weedseg.data.combine import IGNORE_ID, PLANT_ID, combine


def dataset(categories, train_images, image_dir="data/raw/a/images", size=20,
            taxonomy="species"):
    """A converted dataset: `categories` is [(code, role)], `train_images` a list of the codes
    of each training image's plants. val and test hold one image without plants."""
    cats = [{"id": i + 1, "name": code, "supercategory": role}
            for i, (code, role) in enumerate(categories)]
    cats.append({"id": len(cats) + 1, "name": IGNORE_CATEGORY, "supercategory": "ignore"})
    ids = {c["name"]: c["id"] for c in cats}
    info = {"image_dir": image_dir, "min_plant_size_mm": size, "licence": "CC BY 4.0",
            "taxonomy": taxonomy}
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


def test_every_labelled_plant_becomes_a_plant_with_its_own_code():
    a = dataset([("ZEAMX", "crop"), ("POAAN", "weed"), ("1POAG", "weed")],
                [["ZEAMX", "POAAN"], ["ZEAMX", "1POAG"]])
    b = dataset([("BEAVA", "crop"), ("1PLAK", "weed")], [["BEAVA", "1PLAK"]],
                image_dir="data/raw/b/images")
    combined = combine({"a": a, "b": b})
    train = combined["train"]
    assert [c["name"] for c in train["categories"]] == ["plant", IGNORE_CATEGORY]
    got = [(a["eppo"], a["category_id"], a["iscrowd"]) for a in train["annotations"]]
    assert got == [("ZEAMX", PLANT_ID, 0), ("POAAN", PLANT_ID, 0), ("ZEAMX", PLANT_ID, 0),
                   ("1POAG", PLANT_ID, 0), ("BEAVA", PLANT_ID, 0), ("1PLAK", PLANT_ID, 0)]
    assert [a["id"] for a in train["annotations"]] == list(range(1, 7))
    assert [(i["file_name"], i["dataset"]) for i in train["images"]] == [
        ("data/raw/a/images/1.jpg", "a"), ("data/raw/a/images/2.jpg", "a"),
        ("data/raw/b/images/1.jpg", "b")]
    assert [a["image_id"] for a in train["annotations"]] == [1, 1, 2, 2, 3, 3]
    assert train["info"]["image_dir"] == "."
    assert train["info"]["taxonomy"] == "plant" and train["info"]["min_plant_size_mm"] == 20
    assert set(train["info"]["datasets"]) == {"a", "b"}
    assert [len(combined[s]["images"]) for s in ("val", "test")] == [2, 2]


def test_ignore_regions_stay_ignored_and_keep_their_code():
    a = dataset([("ZEAMX", "crop"), ("1PLAK", "weed")], [["ZEAMX", "1PLAK"]])
    ignore_id = a["train"]["categories"][-1]["id"]
    a["train"]["annotations"][1]["iscrowd"] = 1  # an unknown plant the converter left out
    a["train"]["annotations"] += [
        # a plant under the size limit, and an unmapped label
        {"id": 3, "image_id": 1, "iscrowd": 1, "bbox": [0, 0, 1, 1],
         "category_id": ignore_id, "ignored_category_id": 1},
        {"id": 4, "image_id": 1, "iscrowd": 1, "bbox": [0, 0, 5, 5], "category_id": ignore_id}]
    annotations = combine({"a": a})["train"]["annotations"]
    got = [(a.get("eppo"), a["category_id"], a["iscrowd"]) for a in annotations]
    assert got == [("ZEAMX", PLANT_ID, 0), ("1PLAK", IGNORE_ID, 1), ("ZEAMX", IGNORE_ID, 1),
                   (None, IGNORE_ID, 1)]
    assert all("ignored_category_id" not in a for a in annotations)


def test_test_only_datasets_get_a_file_of_their_own_with_all_their_images():
    a = dataset([("ZEAMX", "crop")], [["ZEAMX"]])
    b = dataset([("ZEAMX", "crop"), ("1PLAK", "weed")], [["ZEAMX", "1PLAK"], ["1PLAK"]],
                image_dir="data/raw/b/images")
    c = dataset([("BEAVA", "crop")], [["BEAVA"]], image_dir="data/raw/c")
    c = {"test": c["train"]}  # only a test.json
    combined = combine({"a": a}, {"b": b, "c": c})
    assert list(combined) == ["train", "val", "test", "test_b", "test_c"]
    assert {i["dataset"] for s in ("train", "val", "test") for i in combined[s]["images"]} == {"a"}
    test_b = combined["test_b"]
    # b's train images (2) and its val and test images (1 each)
    assert [i["file_name"] for i in test_b["images"]] == [
        "data/raw/b/images/1.jpg", "data/raw/b/images/2.jpg", "data/raw/b/images/x.jpg",
        "data/raw/b/images/x.jpg"]
    assert [i["id"] for i in test_b["images"]] == [1, 2, 3, 4]
    assert [(a["eppo"], a["image_id"]) for a in test_b["annotations"]] == [
        ("ZEAMX", 1), ("1PLAK", 1), ("1PLAK", 2)]
    assert test_b["info"]["split"] == "test_b" and set(test_b["info"]["datasets"]) == {"b"}
    assert [c["name"] for c in test_b["categories"]] == ["plant", IGNORE_CATEGORY]
    assert [a["eppo"] for a in combined["test_c"]["annotations"]] == ["BEAVA"]


def test_test_only_datasets_are_checked_too():
    a = dataset([("ZEAMX", "crop")], [["ZEAMX"]])
    with pytest.raises(ValueError, match="min_plant_size_mm"):
        combine({"a": a}, {"b": dataset([("ZEAMX", "crop")], [["ZEAMX"]], size=30)})
    with pytest.raises(ValueError, match="both trained on and test only"):
        combine({"a": a}, {"a": a})
    with pytest.raises(ValueError, match="no split file"):
        combine({"a": a}, {"b": {}})


def test_combine_refuses_mixed_sizes_other_taxonomies_and_missing_splits():
    with pytest.raises(ValueError, match="min_plant_size_mm"):
        combine({"a": dataset([("ZEAMX", "crop")], [["ZEAMX"]]),
                 "b": dataset([("ZEAMX", "crop")], [["ZEAMX"]], size=30)})
    with pytest.raises(ValueError, match="taxonomy crop_or_weed2"):
        combine({"a": dataset([("crop", "crop")], [["crop"]], taxonomy="crop_or_weed2")})
    partial = dataset([("ZEAMX", "crop")], [["ZEAMX"]])
    del partial["test"]
    with pytest.raises(ValueError, match="splits"):
        combine({"a": partial})
