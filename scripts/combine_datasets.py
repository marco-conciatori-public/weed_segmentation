"""Combine converted datasets into one training set and choose its classes.
Settings: configs/combine.yaml. See weedseg.data.combine for the class rules.

Writes train.json, val.json, test.json and classes.txt (the class decisions) to
<output_dir>/<dataset names joined by +>/species_min<N>img[_min<size>mm].
"""

from pathlib import Path

from weedseg.config import load_config
from weedseg.data.coco import write_json
from weedseg.data.combine import SPLITS, combine
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.data import load_coco
from weedseg.paths import resolve

CONFIG = "configs/combine.yaml"


def main() -> None:
    config = load_config(CONFIG)
    datasets = {}
    for folder in config["datasets"]:
        name = Path(folder).parent.name
        if name in datasets:
            raise SystemExit(f"Two datasets named {name!r}; combine each dataset once")
        datasets[name] = {split: load_coco(resolve(folder) / f"{split}.json") for split in SPLITS}

    min_images = config["min_train_images"]
    combined, report = combine(datasets, load_taxa(resolve(config["eppo_taxa"])), min_images)
    print("\n".join(report))

    size = combined["train"]["info"]["min_plant_size_mm"]
    folder = f"species_min{min_images}img" + ("" if size is None else f"_min{size:g}mm")
    output_dir = resolve(config["output_dir"]) / "+".join(datasets) / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    for split, coco in combined.items():
        write_json(coco, output_dir / f"{split}.json")
        plants = sum(1 for a in coco["annotations"] if not a["iscrowd"])
        print(f"{split}: {len(coco['images'])} images, {plants} plants -> {output_dir}")


if __name__ == "__main__":
    main()
