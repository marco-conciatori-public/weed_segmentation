"""Convert PhenoBench to COCO (one JSON per split). Settings: configs/phenobench.yaml.

PhenoBench's own split is kept, since it separates crop rows: its train and val images are
our train and val splits, and our test split stays empty (its test labels are not public).
The pipeline (size filter, COCO files) is weedseg.data.convert; scripts/combine_datasets.py
then chooses the classes.
"""

from weedseg.config import load_config
from weedseg.data import phenobench
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/phenobench.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"]) / "PhenoBench"
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    names = phenobench.image_names(raw_dir)
    if not names:
        raise SystemExit(f"No labelled images under {raw_dir}; run scripts/download_datasets.py")

    def records_of(split_names: list[str]):
        return phenobench.iter_records(raw_dir, split_names, settings["mm_per_px_estimate"],
                                       settings["ignore_partial_plants"])

    convert_dataset(names, phenobench.photo_of, lambda name: phenobench.box_labels(raw_dir, name),
                    records_of, label_map, settings, info={
                        "name": "PhenoBench",
                        "source": "https://www.phenobench.org",
                        "licence": "CC BY-NC-SA 4.0; see docs/datasets/phenobench.md",
                        "image_dir": f"{config['raw_dir']}/PhenoBench",  # from the project root
                    }, splits=phenobench.own_splits(names))


if __name__ == "__main__":
    main()
