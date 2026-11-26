"""Merge converted datasets into the plant detector's data (one class, plant).
Settings: configs/combine.yaml. See weedseg.data.combine for what becomes a plant.

Writes train.json, val.json and test.json, and test_<dataset>.json for each test-only dataset,
to <output_dir>/<name>/plant[_min<size>mm], where <name> is the config's `name` or else the
dataset names joined by +: the `data.dir` of configs/detector.yaml.
"""

from collections import Counter
from pathlib import Path

from weedseg.config import load_config
from weedseg.data.coco import write_json
from weedseg.data.combine import PLANT_CATEGORY, PLANT_ID, SPLITS, combine
from weedseg.detector.data import load_coco
from weedseg.paths import resolve

CONFIG = "configs/combine.yaml"


def main(config_path: str = CONFIG) -> Path:
    config = load_config(config_path)
    datasets, test_only, seen = {}, {}, set()
    for folders, target in ((config["datasets"], datasets),
                            (config["test_only"] or [], test_only)):
        for folder in folders:
            dataset = Path(folder).parent.name
            if dataset in seen:
                raise SystemExit(f"Two datasets named {dataset!r}; combine each dataset once")
            seen.add(dataset)
            files = [resolve(folder) / f"{split}.json" for split in SPLITS]
            # a test-only dataset may have only some of the splits (e.g. test.json alone)
            target[dataset] = {path.stem: load_coco(path) for path in files
                               if target is datasets or path.exists()}
    try:
        combined = combine(datasets, test_only)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    size = combined["train"]["info"]["min_plant_size_mm"]
    folder = PLANT_CATEGORY + ("" if size is None else f"_min{size:g}mm")
    output_dir = resolve(config["output_dir"]) / (config["name"] or "+".join(datasets)) / folder
    for split, coco in combined.items():
        write_json(coco, output_dir / f"{split}.json")
        dataset_of = {image["id"]: image["dataset"] for image in coco["images"]}
        images = Counter(dataset_of.values())
        plants = Counter(dataset_of[a["image_id"]] for a in coco["annotations"]
                         if a["category_id"] == PLANT_ID)
        print(f"{split}: {len(coco['images'])} images, {plants.total()} plants -> {output_dir}")
        for dataset in coco["info"]["datasets"]:
            print(f"  {dataset:20} {images[dataset]:6} images {plants[dataset]:7} plants")
    return output_dir


if __name__ == "__main__":
    main()
