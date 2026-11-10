"""Convert CornWeed to COCO (one JSON per split). Settings: configs/cornweed.yaml (convert).

The pipeline (size filter, split by block of frames balanced over species, COCO files) is
weedseg.data.convert; scripts/combine_datasets.py then merges it with the other datasets into
the detector's data.
"""

from weedseg.config import load_config
from weedseg.data import cornweed
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/cornweed.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    try:
        names = cornweed.image_names(raw_dir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    stride = settings["frame_stride"]
    kept = cornweed.thin_frames(names, stride)
    print(f"{len(kept)} of {len(names)} frames kept (one in every {stride})")
    block = settings["group_block"]
    by_width = settings["mm_per_px_by_width"]

    def records_of(split_names: list[str]):
        return cornweed.iter_records(raw_dir, split_names, block, by_width)

    convert_dataset(kept, lambda name: cornweed.group_of(name, block),
                    lambda name: cornweed.box_labels(raw_dir, name), records_of, label_map,
                    settings, info={
                        "name": "CornWeed",
                        "source": "https://zenodo.org/records/7961764",
                        "licence": "CC BY 4.0; see docs/datasets/cornweed.md",
                        # from the project root
                        "image_dir": f"{config['raw_dir']}/{cornweed.ROOT}/data",
                    }, mm_per_px_of=lambda name: cornweed.mm_per_px(raw_dir, name, by_width))


if __name__ == "__main__":
    main()
