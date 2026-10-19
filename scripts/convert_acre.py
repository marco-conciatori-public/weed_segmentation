"""Convert the ACRE Crop-Weed Dataset to COCO (one JSON per split). Settings: configs/acre.yaml.

The pipeline (size filter, split by recording batch balanced over species, COCO files) is
weedseg.data.convert. scripts/combine_datasets.py then chooses the classes.
"""

from weedseg.config import load_config
from weedseg.data import acre
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/acre.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    names = acre.image_names(raw_dir)
    if not names:
        raise SystemExit(f"No annotations found under {raw_dir / 'data'}")

    def records_of(split_names: list[str]):
        return acre.iter_records(raw_dir, split_names, settings["mm_per_px_estimate"])

    convert_dataset(names, acre.batch_of, lambda name: acre.box_labels(raw_dir, name),
                    records_of, label_map, settings, info={
                        "name": "ACRE",
                        "source": "https://doi.org/10.5281/zenodo.8102217",
                        "licence": "CC BY 4.0; see docs/datasets/acre.md",
                        "image_dir": f"{config['raw_dir']}/data",  # from the project root
                    })


if __name__ == "__main__":
    main()
