"""Convert OPPD to COCO (one JSON per split). Settings: configs/oppd.yaml.

OPPD is labelled with boxes only, so its images are marked box_only: they reach the species
classifier, not Mask R-CNN. The pipeline (size filter, split by tray balanced over species,
COCO files) is weedseg.data.convert; scripts/combine_datasets.py then chooses the classes.
"""

from weedseg.config import load_config
from weedseg.data import oppd
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/oppd.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    names = oppd.image_names(raw_dir)
    if not names:
        raise SystemExit(f"No annotations under {raw_dir / 'ds0' / 'ann'}; "
                         "run scripts/download_datasets.py")

    def records_of(split_names: list[str]):
        return oppd.iter_records(raw_dir, split_names, settings["mm_per_px_estimate"])

    convert_dataset(names, oppd.tray_of, lambda name: oppd.box_labels(raw_dir, name),
                    records_of, label_map, settings, info={
                        "name": "OPPD",
                        "source": "https://vision.eng.au.dk/open-plant-phenotyping-database/ "
                                  "(Dataset Ninja's copy)",
                        "licence": "CC BY-NC-SA 4.0; see docs/datasets/oppd.md",
                        "image_dir": f"{config['raw_dir']}/ds0/img",  # from the project root
                    }, box_only=True)


if __name__ == "__main__":
    main()
