"""Convert CropAndWeed to COCO (one JSON per split). Settings: configs/cropandweed.yaml (convert).

Order: size filter -> split by session, balanced over species -> classes from the training
images (with min_train_images) -> COCO files, small plants as ignore regions.
"""

from collections import Counter

from tqdm import tqdm

from weedseg.config import load_config
from weedseg.data import cropandweed
from weedseg.data.coco import (
    LabelMap,
    ignore_small_plants,
    is_small,
    split_by_group,
    to_coco,
    write_json,
)
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/cropandweed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    min_size_mm, mm_per_px = settings["min_plant_size_mm"], settings["mm_per_px_estimate"]
    if min_size_mm is not None and mm_per_px is None:
        raise SystemExit("min_plant_size_mm needs mm_per_px_estimate")

    names = cropandweed.image_names(raw_dir, settings["keep_negative_images"],
                                    settings["require_image_file"])
    if not names:
        raise SystemExit(f"No images found under {raw_dir}; run scripts/download_cropandweed.py")
    # Source labels of the plants that pass the size filter, per image.
    plant_labels = {name: {label for label, width, height in cropandweed.box_labels(raw_dir, name)
                           if not is_small(width, height, mm_per_px, min_size_mm)}
                    for name in names}
    if not settings["keep_negative_images"]:  # also drop images left without plants
        names = [name for name in names if plant_labels[name]]

    def species_of(name: str) -> set[str]:
        return {label_map.rows[label]["eppo"] for label in plant_labels[name]} - {""}

    split = settings["split"]
    splits = split_by_group(names, cropandweed.session_of, split["fractions"], split["seed"],
                            labels_of=species_of)

    report = []
    if settings["min_train_images"] is not None:
        report = label_map.keep_common_species([plant_labels[name] for name in splits["train"]],
                                               settings["min_train_images"])
        print("\n".join(report))
    folder = label_map.taxonomy + ("" if min_size_mm is None else f"_min{min_size_mm:g}mm")
    output_dir = resolve(settings["output_dir"]) / folder
    if report:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "classes.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    for split_name, split_names in splits.items():
        records = cropandweed.iter_records(raw_dir, split_names, settings["instance_masks"],
                                           settings["unboxed_min_area"], mm_per_px)
        info = {
            "description": f"CropAndWeed, {split_name} split, taxonomy {label_map.taxonomy}",
            "source": "https://github.com/cropandweed/cropandweed-dataset",
            "licence": "Non-commercial use only; see docs/datasets/cropandweed.md",
            "image_dir": f"{config['raw_dir']}/images",  # relative to the project root
            "split": split_name,
            "split_seed": split["seed"],
            "min_plant_size_mm": min_size_mm,
        }
        coco = to_coco(tqdm(records, total=len(split_names), desc=split_name), label_map, info)
        coco = ignore_small_plants(coco, min_size_mm)
        path = output_dir / f"{split_name}.json"
        write_json(coco, path)

        counts = Counter(a["category_id"] for a in coco["annotations"] if not a["iscrowd"])
        sessions = len({image["group"] for image in coco["images"]})
        print(f"{split_name}: {len(coco['images'])} images from {sessions} sessions, "
              f"{sum(counts.values())} instances in {len(counts)} categories -> {path}")


if __name__ == "__main__":
    main()
