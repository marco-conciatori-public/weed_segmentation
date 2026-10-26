"""Convert GROUNDBASED_WEED to COCO (one JSON per split). Settings: configs/groundbased_weed.yaml.

GROUNDBASED_WEED is labelled with boxes only, so its images are marked box_only: they reach
the species classifier, not Mask R-CNN. The pipeline (size filter, split by block of camera
numbers balanced over species, COCO files) is weedseg.data.convert;
scripts/combine_datasets.py then chooses the classes.
"""

from weedseg.config import load_config
from weedseg.data import groundbased_weed
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/groundbased_weed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    names = groundbased_weed.image_names(raw_dir)
    if not names:
        raise SystemExit(f"No annotations under {raw_dir}/TOMATO_*/Annotations; "
                         "run scripts/download_datasets.py")
    block = settings["group_block"]

    def records_of(split_names: list[str]):
        return groundbased_weed.iter_records(raw_dir, split_names, block,
                                             settings["mm_per_px_estimate"])

    convert_dataset(names, lambda name: groundbased_weed.group_of(name, block),
                    lambda name: groundbased_weed.box_labels(raw_dir, name),
                    records_of, label_map, settings, info={
                        "name": "GROUNDBASED_WEED",
                        "source": "https://doi.org/10.20350/digitalCSIC/17475",
                        "licence": "ODbL 1.0; see docs/datasets/groundbased_weed.md",
                        "image_dir": config["raw_dir"],  # from the project root
                    }, box_only=True)


if __name__ == "__main__":
    main()
