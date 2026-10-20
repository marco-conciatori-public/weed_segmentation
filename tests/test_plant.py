from weedseg.data.plant import PLANT_CATEGORY, to_plant_only


def annotation(id_, category_id, bbox, iscrowd=0, **extra):
    return {"id": id_, "image_id": 1, "category_id": category_id, "bbox": bbox,
            "iscrowd": iscrowd, **extra}


def test_plants_merge_and_only_hidden_known_plants_leave_ignore_regions():
    coco = {
        "info": {"min_plant_size_mm": 20},
        "images": [{"id": 1, "mm_per_px": 0.5}],
        "categories": [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "SETVI"},
                       {"id": 3, "name": "other_plant"}, {"id": 4, "name": "ignore"}],
        "annotations": [
            annotation(1, 1, [0, 0, 60, 60]),  # maize
            annotation(2, 3, [0, 0, 50, 50]),  # other_plant
            annotation(3, 4, [0, 0, 60, 60], 1, eppo="1SETG"),  # genus hidden by rule 3
            annotation(4, 4, [0, 0, 20, 20], 1, eppo="SETVI"),  # 10 mm: too small
            annotation(5, 4, [0, 0, 80, 80], 1),  # unidentified vegetation
            annotation(6, 2, [0, 0, 60, 60], 1, eppo="SETVI"),  # converter ignore, own class
        ],
    }
    plants = to_plant_only(coco)
    assert [c["name"] for c in plants["categories"]] == [PLANT_CATEGORY, "ignore"]
    assert [(a["category_id"], a["iscrowd"]) for a in plants["annotations"]] == [
        (1, 0), (1, 0), (1, 0), (2, 1), (2, 1), (2, 1)]
    assert plants["annotations"][2]["eppo"] == "1SETG"  # other fields kept
    assert coco["annotations"][2]["iscrowd"] == 1  # input untouched
