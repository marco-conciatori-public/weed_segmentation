"""Convert CropAndWeed to COCO (one JSON per split). Settings: configs/cropandweed.yaml (convert)."""

from collections import Counter

from tqdm import tqdm

from weedseg.config import load_config
from weedseg.data import cropandweed
from weedseg.data.coco import LabelMap, split_by_group, to_coco, write_json
from weedseg.paths import resolve

CONFIG = "configs/cropandweed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    output_dir = resolve(settings["output_dir"]) / settings["taxonomy"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"])

    names = cropandweed.image_names(raw_dir, settings["keep_negative_images"],
                                    settings["require_image_file"])
    if not names:
        raise SystemExit(f"No images found under {raw_dir}; run scripts/download_cropandweed.py")
    split = settings["split"]
    splits = split_by_group(names, cropandweed.session_of, split["fractions"], split["seed"])

    for split_name, split_names in splits.items():
        records = cropandweed.iter_records(raw_dir, split_names, settings["instance_masks"],
                                           settings["unboxed_min_area"],
                                           settings["mm_per_px_estimate"])
        info = {
            "description": f"CropAndWeed, {split_name} split, taxonomy {settings['taxonomy']}",
            "source": "https://github.com/cropandweed/cropandweed-dataset",
            "licence": "Non-commercial use only; see docs/datasets/cropandweed.md",
            "image_dir": f"{config['raw_dir']}/images",  # relative to the project root
            "split": split_name,
            "split_seed": split["seed"],
        }
        coco = to_coco(tqdm(records, total=len(split_names), desc=split_name), label_map, info)
        path = output_dir / f"{split_name}.json"
        write_json(coco, path)

        counts = Counter(a["category_id"] for a in coco["annotations"] if not a["iscrowd"])
        sessions = len({image["group"] for image in coco["images"]})
        print(f"{split_name}: {len(coco['images'])} images from {sessions} sessions, "
              f"{sum(counts.values())} instances in {len(counts)} categories -> {path}")


if __name__ == "__main__":
    main()
