"""The conversion pipeline shared by all datasets (see docs/data-format.md).

Order: size filter -> split by group, balanced over species (or the dataset's own split) ->
COCO files, small plants as ignore regions. A dataset module only provides its image names, their groups, the boxes of
each image (cheap, for the size filter and the split) and the full records.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from weedseg.data.coco import (
    ImageRecord,
    LabelMap,
    ignore_small_plants,
    is_small,
    split_by_group,
    to_coco,
    write_json,
)
from weedseg.paths import resolve


def convert_dataset(names: list[str], group_of: Callable[[str], str],
                    boxes_of: Callable[[str], list[tuple[str, float, float]]],
                    records_of: Callable[[list[str]], Iterator[ImageRecord]],
                    label_map: LabelMap, settings: dict[str, Any],
                    info: dict[str, Any], box_only: bool = False,
                    splits: dict[str, list[str]] | None = None) -> Path:
    """Convert a dataset and write <output_dir>/<taxonomy>[_min<size>mm]/<split>.json.

    `boxes_of(name)` gives (source label, width, height) in px of each plant of an image;
    `records_of(names)` yields their ImageRecords. `settings` is the dataset config's
    `convert` section (min_plant_size_mm, mm_per_px_estimate, keep_negative_images, split,
    output_dir); `info` (name, source, licence, image_dir) goes into every file's info.
    `box_only` for a dataset labelled with boxes only, without masks: its images are marked
    (weedseg.data.coco.BOX_ONLY) and reach the species classifier only. `splits` keeps the
    dataset's own split (image names per split; `settings` then needs no `split`), for a
    dataset whose split is cleaner than ours, e.g. separated in space.
    """
    min_size_mm, mm_per_px = settings["min_plant_size_mm"], settings["mm_per_px_estimate"]
    if min_size_mm is not None and mm_per_px is None:
        raise SystemExit("min_plant_size_mm needs mm_per_px_estimate")
    # Source labels of the plants that pass the size filter, per image.
    plant_labels = {name: {label for label, width, height in boxes_of(name)
                           if not is_small(width, height, mm_per_px, min_size_mm)}
                    for name in names}
    if not settings["keep_negative_images"]:  # also drop images left without plants
        names = [name for name in names if plant_labels[name]]

    def species_of(name: str) -> set[str]:
        return {label_map.rows[label]["eppo"] for label in plant_labels[name]} - {""}

    if splits is None:
        seed = settings["split"]["seed"]
        splits = split_by_group(names, group_of, settings["split"]["fractions"], seed,
                                labels_of=species_of)
    else:  # the dataset's own split, without the images dropped above
        seed, kept = None, set(names)
        splits = {name: [n for n in split_names if n in kept]
                  for name, split_names in splits.items()}

    folder = label_map.taxonomy + ("" if min_size_mm is None else f"_min{min_size_mm:g}mm")
    output_dir = resolve(settings["output_dir"]) / folder
    for split_name, split_names in splits.items():
        split_info = {
            **info,
            "description": f"{info['name']}, {split_name} split, taxonomy {label_map.taxonomy}",
            "split": split_name,
            "split_seed": seed,
            "min_plant_size_mm": min_size_mm,
        }
        records = tqdm(records_of(split_names), total=len(split_names), desc=split_name)
        coco = ignore_small_plants(to_coco(records, label_map, split_info, box_only), min_size_mm)
        path = output_dir / f"{split_name}.json"
        write_json(coco, path)

        counts = Counter(a["category_id"] for a in coco["annotations"] if not a["iscrowd"])
        groups = len({image["group"] for image in coco["images"]})
        print(f"{split_name}: {len(coco['images'])} images from {groups} groups, "
              f"{sum(counts.values())} instances in {len(counts)} categories -> {path}")
    return output_dir
