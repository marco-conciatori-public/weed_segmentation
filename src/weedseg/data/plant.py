"""Species COCO files -> one `plant` class, for the detector of the two-stage pipeline.

Every labelled plant becomes `plant`, whatever its species. Ignore regions stay ignore
regions, except plants hidden only by the choice of classes: combining datasets turns a
genus-only label into an ignore region when one of its species is a class (rule 3 in
weedseg.data.combine), but for a detector it is a plant like any other. Those are told apart
by what they carry: the `ignore` category, an EPPO code (unidentified vegetation has none)
and a size at least the minimum (smaller plants are ignored on purpose). Ignore regions that
kept a species category come from the source converters (e.g. a mask that could not be
derived) and stay ignored.
"""

from typing import Any

from weedseg.data.coco import IGNORE_CATEGORY, is_small

PLANT_CATEGORY = "plant"


def is_hidden_plant(annotation: dict[str, Any], ignore_id: int, mm_per_px: float | None,
                    min_size_mm: float | None) -> bool:
    """Whether an ignore region is a known plant of at least the minimum size."""
    if annotation["category_id"] != ignore_id or not annotation.get("eppo"):
        return False
    if min_size_mm is None:
        return True
    if mm_per_px is None:
        return False  # size unknown: keep it ignored
    return not is_small(*annotation["bbox"][2:], mm_per_px, min_size_mm)


def to_plant_only(coco: dict[str, Any]) -> dict[str, Any]:
    """Copy of a species COCO dict with the categories `plant` (id 1) and `ignore` (id 2)."""
    ignore_ids = {c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY}
    ignore_id = next(iter(ignore_ids), None)
    min_size_mm = coco["info"].get("min_plant_size_mm")
    mm_per_px = {i["id"]: i.get("mm_per_px") for i in coco["images"]}
    annotations = []
    for annotation in coco["annotations"]:
        crowd = bool(annotation.get("iscrowd")) or annotation["category_id"] in ignore_ids
        plant = not crowd or is_hidden_plant(annotation, ignore_id, mm_per_px[annotation["image_id"]],
                                             min_size_mm)
        annotations.append({**annotation, "category_id": 1 if plant else 2,
                            "iscrowd": 0 if plant else 1})
    categories = [{"id": 1, "name": PLANT_CATEGORY, "supercategory": PLANT_CATEGORY},
                  {"id": 2, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY}]
    info = {**coco["info"], "taxonomy": PLANT_CATEGORY,
            "description": f"{coco['info'].get('description', '')} -> one plant class"}
    return {**coco, "info": info, "categories": categories, "annotations": annotations}
