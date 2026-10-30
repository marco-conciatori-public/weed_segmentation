"""Combine converted datasets into one training set and choose its classes.
Settings: configs/combine.yaml. See weedseg.data.combine for the class rules.

Writes train.json, val.json, test.json and classes.txt (the class decisions) to
<output_dir>/<name>/species_min<N>img[_min<size>mm], where <name> is the config's `name`
or else the dataset names joined by +. Each of the config's `variants` (the datasets plus
some more, for classifier variants trained on other data) is combined the same way into a
folder of its own.
"""

from pathlib import Path
from typing import Any

from weedseg.config import load_config
from weedseg.data.coco import BOX_ONLY, write_json
from weedseg.data.combine import SPLITS, combine
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.data import load_coco
from weedseg.paths import resolve

CONFIG = "configs/combine.yaml"


def combinations(config: dict[str, Any]) -> list[tuple[str | None, list[str]]]:
    """(name, dataset folders) of the config's combination and of each of its variants."""
    found = [(config["name"], config["datasets"])]
    for variant in config["variants"] or []:
        found.append((variant["name"], config["datasets"] + variant["add"]))
    return found


def write_combination(name: str | None, folders: list[str], config: dict[str, Any]) -> Path:
    """Combine the converted dataset folders; the output folder."""
    datasets = {}
    for folder in folders:
        dataset = Path(folder).parent.name
        if dataset in datasets:
            raise SystemExit(f"Two datasets named {dataset!r}; combine each dataset once")
        datasets[dataset] = {split: load_coco(resolve(folder) / f"{split}.json")
                             for split in SPLITS}

    min_images = config["min_train_images"]
    combined, report = combine(datasets, load_taxa(resolve(config["eppo_taxa"])), min_images)
    print("\n".join(report))

    size = combined["train"]["info"]["min_plant_size_mm"]
    folder = f"species_min{min_images}img" + ("" if size is None else f"_min{size:g}mm")
    output_dir = resolve(config["output_dir"]) / (name or "+".join(datasets)) / folder
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text(
        f"Datasets: {', '.join(datasets)}\n" + "\n".join(report) + "\n", encoding="utf-8")
    for split, coco in combined.items():
        write_json(coco, output_dir / f"{split}.json")
        plants = sum(1 for a in coco["annotations"] if not a["iscrowd"])
        box_only = sum(1 for image in coco["images"] if image.get(BOX_ONLY))
        print(f"{split}: {len(coco['images'])} images"
              + (f" ({box_only} with boxes only: classifier only)" if box_only else "")
              + f", {plants} plants -> {output_dir}")
    return output_dir


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    for name, folders in combinations(config):
        print(f"===== {name or '+'.join(Path(folder).parent.name for folder in folders)}")
        write_combination(name, folders, config)


if __name__ == "__main__":
    main()
