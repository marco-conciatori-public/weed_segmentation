"""Convert WeedMaize to COCO (one JSON per split). Settings: configs/weedmaize.yaml (convert).

The photos come cut into heavily overlapping tiles; only a spread-out set of tiles of each
photo is kept (weedseg.data.weedmaize.select_tiles), so a plant isn't learned many times
over. The pipeline (size filter, split by block of photos balanced over species, COCO files)
is weedseg.data.convert; scripts/combine_datasets.py then merges it with the other datasets
into the detector's data.
"""

from weedseg.config import load_config
from weedseg.data import weedmaize
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/weedmaize.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    try:
        names = weedmaize.image_names(raw_dir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    kept = weedmaize.select_tiles(raw_dir, names, settings["max_tile_overlap"])
    photos = len({weedmaize.parse(name)[1] for name in names})
    print(f"{len(kept)} of {len(names)} tiles kept ({photos} photos), at most "
          f"{settings['max_tile_overlap']:.0%} of a tile shared with another")
    block = settings["group_block"]
    labelling = weedmaize.Labelling(settings["duplicate_iou"], settings["centres"],
                                    settings["centre_ioa"])

    def records_of(split_names: list[str]):
        return weedmaize.iter_records(raw_dir, split_names, block, labelling,
                                      settings["mm_per_px_estimate"])

    convert_dataset(kept, lambda name: weedmaize.group_of(name, block),
                    lambda name: weedmaize.box_labels(raw_dir, name, labelling),
                    records_of, label_map, settings, info={
                        "name": "WeedMaize",
                        "source": "https://zenodo.org/records/5106795",
                        "licence": "CC BY 4.0; see docs/datasets/weedmaize.md",
                        "image_dir": config["raw_dir"],  # from the project root
                    })


if __name__ == "__main__":
    main()
