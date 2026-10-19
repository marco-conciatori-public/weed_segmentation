"""Convert CropAndWeed to COCO (one JSON per split). Settings: configs/cropandweed.yaml (convert).

The pipeline (size filter, split by session balanced over species, COCO files) is
weedseg.data.convert. With taxonomy species, scripts/combine_datasets.py then chooses the
classes.
"""

from weedseg.config import load_config
from weedseg.data import cropandweed
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/cropandweed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    names = cropandweed.image_names(raw_dir, settings["keep_negative_images"],
                                    settings["require_image_file"])
    if not names:
        raise SystemExit(f"No images found under {raw_dir}; run scripts/download_cropandweed.py")

    def records_of(split_names: list[str]):
        return cropandweed.iter_records(raw_dir, split_names, settings["instance_masks"],
                                        settings["unboxed_min_area"],
                                        settings["mm_per_px_estimate"])

    convert_dataset(names, cropandweed.session_of,
                    lambda name: cropandweed.box_labels(raw_dir, name), records_of,
                    label_map, settings, info={
                        "name": "CropAndWeed",
                        "source": "https://github.com/cropandweed/cropandweed-dataset",
                        "licence": "Non-commercial use only; see docs/datasets/cropandweed.md",
                        "image_dir": f"{config['raw_dir']}/images",  # from the project root
                    })


if __name__ == "__main__":
    main()
