"""Dataset-independent conversion to our internal COCO format (see docs/data-format.md).

A dataset module only has to yield `ImageRecord`s with source labels. This module maps the
labels to categories through a label-map CSV, splits images by group and writes COCO JSON.
"""

import csv
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

IGNORE_CATEGORY = "ignore"


@dataclass
class Instance:
    """One annotated plant, with its label as written in the source dataset."""

    source_label: str
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None  # bool, full image size
    stem_xy: tuple[float, float] | None = None
    ignore: bool = False  # keep as an ignore region (iscrowd) whatever its category
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRecord:
    file_name: str  # relative to the dataset's image directory
    width: int
    height: int
    group: str  # images of one group always go to the same split (e.g. a recording session)
    instances: list[Instance] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


class LabelMap:
    """Maps source labels to categories, read from a CSV with one row per source label.

    Required columns: `source_label`, `role` (crop/weed) and one column per taxonomy whose
    values are category names. An empty taxonomy value marks labels to be ignored. Any other
    columns (common name, scientific name, ...) are kept for the category metadata.
    """

    def __init__(self, path: Path, taxonomy: str):
        with open(path, newline="", encoding="utf-8") as f:
            self.rows = {row["source_label"]: row for row in csv.DictReader(f)}
        first = next(iter(self.rows.values()))
        if taxonomy not in first:
            raise KeyError(f"Taxonomy column {taxonomy!r} not in {path}")
        self.taxonomy = taxonomy

    def category(self, source_label: str) -> str:
        """Return the category name, or IGNORE_CATEGORY for unmapped labels."""
        if source_label not in self.rows:
            raise KeyError(f"Source label {source_label!r} missing from the label map")
        return self.rows[source_label][self.taxonomy] or IGNORE_CATEGORY

    def categories(self) -> list[dict[str, Any]]:
        """COCO categories of the selected taxonomy, ignore category last."""
        members: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.rows.values():
            if row[self.taxonomy]:
                members[row[self.taxonomy]].append(row)
        categories = []
        for name in sorted(members, key=lambda n: (_role(members[n]) != "crop", n)):
            rows = members[name]
            categories.append({
                "id": len(categories) + 1,
                "name": name,
                "supercategory": _role(rows),
                "source_labels": [row["source_label"] for row in rows],
                "common_names": sorted({row.get("common_name", "") for row in rows} - {""}),
                "scientific_names": sorted({row.get("scientific_name", "") for row in rows} - {""}),
            })
        categories.append({"id": len(categories) + 1, "name": IGNORE_CATEGORY,
                           "supercategory": IGNORE_CATEGORY, "source_labels": []})
        return categories


def _role(rows: list[dict[str, str]]) -> str:
    roles = {row["role"] for row in rows}
    return roles.pop() if len(roles) == 1 else "mixed"


def split_by_group[T](items: list[T], group_of: Callable[[T], str],
                      fractions: dict[str, float], seed: int) -> dict[str, list[T]]:
    """Assign whole groups to splits so each split gets about its fraction of the items."""
    by_group: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_group[group_of(item)].append(item)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)

    total = sum(fractions.values())
    targets = {name: fraction / total * len(items) for name, fraction in fractions.items()}
    splits: dict[str, list[T]] = {name: [] for name in fractions}
    for group in groups:
        # Give the group to the split furthest below its target share.
        name = max(splits, key=lambda n: targets[n] - len(splits[n]))
        splits[name].extend(by_group[group])
    return splits


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Compressed COCO RLE of a bool mask, JSON-serialisable."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def to_coco(records: Iterable[ImageRecord], label_map: LabelMap,
            info: dict[str, Any]) -> dict[str, Any]:
    """Build a COCO dict. Ignored instances, and those mapped to the ignore category, get
    `iscrowd = 1`, so COCO evaluation neither rewards nor penalises detections on them."""
    categories = label_map.categories()
    category_ids = {c["name"]: c["id"] for c in categories}
    images, annotations = [], []
    for record in records:
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": record.file_name, "width": record.width,
                       "height": record.height, "group": record.group, **record.attributes})
        for instance in record.instances:
            category = label_map.category(instance.source_label)
            x0, y0, x1, y1 = instance.bbox_xyxy
            annotation = {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_ids[category],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": float((x1 - x0) * (y1 - y0)),
                "iscrowd": int(instance.ignore or category == IGNORE_CATEGORY),
                "source_label": instance.source_label,
                **instance.attributes,
            }
            if instance.mask is not None:
                annotation["segmentation"] = encode_mask(instance.mask)
                annotation["area"] = float(instance.mask.sum())
            if instance.stem_xy is not None:
                annotation["stem"] = list(instance.stem_xy)
            annotations.append(annotation)
    return {"info": info, "images": images, "annotations": annotations, "categories": categories}


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
