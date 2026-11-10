"""Convert DeepSeedling to COCO (one JSON per split). Settings: configs/deepseedling.yaml
(convert).

The pipeline (size filter, split by block of frames balanced over species, COCO files) is
weedseg.data.convert; scripts/combine_datasets.py then merges it with the other datasets into
the detector's data.
"""

from weedseg.config import load_config
from weedseg.data import deepseedling
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/deepseedling.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    sets = settings["sets"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    try:
        names = deepseedling.image_names(raw_dir, sets)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    kept = deepseedling.thin_frames(raw_dir, names, sets)
    for set_name, set_settings in sets.items():
        in_set = [n for n in names if n.startswith(f"{set_name}/")]
        kept_in_set = [n for n in kept if n.startswith(f"{set_name}/")]
        print(f"{set_name}: {len(kept_in_set)} of {len(in_set)} labelled frames kept "
              f"(one in every {set_settings['frame_stride']})")

    convert_dataset(kept, lambda name: deepseedling.group_of(raw_dir, name, sets),
                    lambda name: deepseedling.box_labels(raw_dir, name, sets),
                    lambda split_names: deepseedling.iter_records(raw_dir, split_names, sets),
                    label_map, settings, info={
                        "name": "DeepSeedling",
                        "source": "https://figshare.com/s/616956f8633c17ceae9b",
                        "licence": "GPL 3.0+; see docs/datasets/deepseedling.md",
                        # from the project root
                        "image_dir": config["raw_dir"],
                    }, mm_per_px_of=lambda name: deepseedling.mm_per_px(name, sets))


if __name__ == "__main__":
    main()
