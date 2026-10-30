"""Convert datasets downloaded from Weed-AI (WeedCOCO format) to COCO, one JSON per split.
Settings: configs/weedai.yaml lists the dataset configs to convert (configs/lincolnbeet.yaml,
...), each with its own `convert` section.

Weed-AI's datasets are labelled with boxes only, so their images are marked box_only: they
reach the species classifier, not Mask R-CNN. The pipeline (size filter, split by group
balanced over species, COCO files) is weedseg.data.convert, the reader weedseg.data.weedai;
scripts/combine_datasets.py then chooses the classes. See docs/datasets/weedai.md.
"""

from pathlib import Path

from weedseg.config import load_config
from weedseg.data import weedai
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/weedai.yaml"


def convert(config_path: str) -> Path:
    """Convert the dataset that `config_path` describes; its output folder."""
    config = load_config(config_path)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    try:
        names = weedai.image_names(raw_dir, settings["parts"])
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    if settings["drop_duplicate_images"]:
        repeated = weedai.duplicates(raw_dir, names)
        print(f"{len(repeated)} images are copies of earlier ones, left out"
              + (f", e.g. {repeated[0]}" if repeated else ""))
        names = sorted(set(names) - set(repeated))
    grouping = weedai.Grouping(settings["group_pattern"], settings["group_block"])
    scale = weedai.Scale(settings["mm_per_px_estimate"], settings["footprint_mm_estimate"])
    drop = frozenset(settings["drop_labels"])

    def records_of(split_names: list[str]):
        return weedai.iter_records(raw_dir, split_names, grouping, scale, drop)

    return convert_dataset(
        names, lambda name: grouping.group_of(weedai.source_name(name)),
        lambda name: weedai.box_labels(raw_dir, name, drop), records_of, label_map, settings,
        info={"name": config["name"], "source": config["source"], "licence": config["licence"],
              "image_dir": config["raw_dir"]},  # from the project root
        box_only=True, mm_per_px_of=lambda name: weedai.mm_per_px(raw_dir, name, scale))


def main(config_path: str = CONFIG) -> None:
    for dataset_config in load_config(config_path)["datasets"]:
        print(f"===== {dataset_config}")
        convert(dataset_config)


if __name__ == "__main__":
    main()
