"""Merge converted datasets into the plant detector's data: one class, `plant`.

Each dataset is first converted on its own (scripts/convert_*.py with taxonomy `species`: one
category per EPPO code, small plants as ignore regions, its own split by group). Merging keeps
each dataset's split and makes every labelled plant a `plant`, whatever its label: a species, a
genus only, or a plant of unknown species (1PLAK). Ignore regions stay ignore regions: the
plants the converter left out (unidentified vegetation, too small, partly visible, untrusted,
plants without a box).

Every annotation keeps the EPPO code of its own label in `eppo` (when it has one), so the
evaluation can tell which species the detector misses.

Only datasets whose every plant has a box (or an ignore region) belong here: an unboxed plant
is taught as background. See docs/detector.md for which datasets qualify.
"""

from typing import Any

from weedseg.data.coco import IGNORE_CATEGORY

SPLITS = ("train", "val", "test")
PLANT_CATEGORY = "plant"
PLANT_ID, IGNORE_ID = 1, 2
CATEGORIES = [{"id": PLANT_ID, "name": PLANT_CATEGORY, "supercategory": PLANT_CATEGORY},
              {"id": IGNORE_ID, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY}]


def combine(datasets: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Merge converted datasets ({name: {split: COCO dict}}) into one COCO dict per split, with
    the categories `plant` and `ignore`.

    Image file names become paths relative to the project root (`info.image_dir` is "."),
    since each dataset keeps its images in its own folder; each image gets its `dataset`.
    """
    sizes = {name: {coco["info"].get("min_plant_size_mm") for coco in splits.values()}
             for name, splits in datasets.items()}
    if len({size for values in sizes.values() for size in values}) > 1:
        raise ValueError(f"The datasets were converted with different min_plant_size_mm: {sizes}")
    codes = {}  # (dataset, category id) -> its EPPO code; None for the ignore category
    for name, splits in datasets.items():
        if set(splits) != set(SPLITS):
            raise ValueError(f"{name} must have the splits {', '.join(SPLITS)}")
        taxonomy = splits["train"]["info"].get("taxonomy")
        if taxonomy != "species":
            raise ValueError(f"{name} was converted with taxonomy {taxonomy}, not species: "
                             "convert it again with taxonomy species")
        for category in splits["train"]["categories"]:
            codes[name, category["id"]] = (None if category["name"] == IGNORE_CATEGORY
                                           else category["name"])

    combined = {}
    for split in SPLITS:
        out_images, out_annotations = [], []
        for name, splits in datasets.items():
            coco = splits[split]
            image_dir = coco["info"]["image_dir"].rstrip("/")
            new_image_id = {}
            for image in coco["images"]:
                new_image_id[image["id"]] = len(out_images) + 1
                out_images.append({**image, "id": len(out_images) + 1, "dataset": name,
                                   "file_name": f"{image_dir}/{image['file_name']}"})
            for annotation in coco["annotations"]:
                # a plant under the size limit keeps its own category in ignored_category_id
                code = codes[name, annotation.get("ignored_category_id",
                                                  annotation["category_id"])]
                plant = not annotation.get("iscrowd")
                new = {key: value for key, value in annotation.items()
                       if key != "ignored_category_id"}
                new.update(id=len(out_annotations) + 1,
                           image_id=new_image_id[annotation["image_id"]],
                           category_id=PLANT_ID if plant else IGNORE_ID, iscrowd=int(not plant))
                if code:
                    new["eppo"] = code
                out_annotations.append(new)
        combined[split] = {
            "info": {
                "description": f"{' + '.join(datasets)}, {split} split, one class (plant)",
                "datasets": {name: splits[split]["info"] for name, splits in datasets.items()},
                "licences": sorted({splits[split]["info"].get("licence", "unknown")
                                    for splits in datasets.values()}),
                "image_dir": ".",
                "split": split,
                "taxonomy": PLANT_CATEGORY,
                "min_plant_size_mm": next(iter(next(iter(sizes.values())))),
            },
            "images": out_images,
            "annotations": out_annotations,
            "categories": CATEGORIES,
        }
    return combined
