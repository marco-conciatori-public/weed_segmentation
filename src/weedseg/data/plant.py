"""Species COCO files -> one `plant` class, for the detector of the two-stage pipeline.

Every labelled plant becomes `plant`, whatever its species. Ignore regions stay ignore
regions, except plants hidden only by the choice of classes: combining datasets turns a
genus-only label into an ignore region when one of its species is a class (rule 3 in
weedseg.data.combine), and likewise a plant of unknown species (Plantae, 1PLAK), but for a
detector they are plants like any other. Combining marks them `hidden_plant`. Every other
ignore region stays ignored: plants the converter left out (unidentified, too small,
untrusted, without a usable outline) carry no mark, even when they have an EPPO code.
"""

from typing import Any

from weedseg.data.coco import IGNORE_CATEGORY

PLANT_CATEGORY = "plant"


def to_plant_only(coco: dict[str, Any]) -> dict[str, Any]:
    """Copy of a species COCO dict with the categories `plant` (id 1) and `ignore` (id 2)."""
    if "datasets" in coco["info"] and "hidden_plants" not in coco["info"]:
        raise ValueError("This combined dataset was made before hidden plants were marked: "
                         "re-run scripts/combine_datasets.py")
    ignore_ids = {c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY}
    annotations = []
    for annotation in coco["annotations"]:
        crowd = bool(annotation.get("iscrowd")) or annotation["category_id"] in ignore_ids
        plant = not crowd or bool(annotation.get("hidden_plant"))
        annotations.append({**annotation, "category_id": 1 if plant else 2,
                            "iscrowd": 0 if plant else 1})
    categories = [{"id": 1, "name": PLANT_CATEGORY, "supercategory": PLANT_CATEGORY},
                  {"id": 2, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY}]
    info = {**coco["info"], "taxonomy": PLANT_CATEGORY,
            "description": f"{coco['info'].get('description', '')} -> one plant class"}
    return {**coco, "info": info, "categories": categories, "annotations": annotations}
