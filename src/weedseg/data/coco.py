"""Dataset-independent conversion to our internal COCO format (see docs/data-format.md).

A dataset module only has to yield `ImageRecord`s with source labels. This module maps the
labels to categories through a label-map CSV, splits images by group and writes COCO JSON.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from weedseg.eppo import PARENT_RANKS

IGNORE_CATEGORY = "ignore"
# Plants of a known species too rare to be a class of their own (see
# LabelMap.keep_common_species); at inference they are handed to a generic model.
OTHER_CATEGORY = "other_plant"


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

    The taxonomies `genus`, `family` and `order` need no column: they come from the `eppo`
    column and the EPPO taxa table (`taxa`, see weedseg.eppo.load_taxa). They group weeds
    only; crops keep their own EPPO code, since a crop must never share a class with a
    weed (by family, maize would join the grass weeds).
    """

    def __init__(self, path: Path, taxonomy: str,
                 taxa: dict[str, dict[str, str]] | None = None):
        with open(path, newline="", encoding="utf-8") as f:
            self.rows = {row["source_label"]: row for row in csv.DictReader(f)}
        first = next(iter(self.rows.values()))
        if taxonomy not in first and taxonomy in PARENT_RANKS:
            if taxa is None:
                raise ValueError(f"Taxonomy {taxonomy!r} needs the EPPO taxa table")
            crops = {row["eppo"] for row in self.rows.values() if row["role"] == "crop"}
            for row in self.rows.values():
                group = taxa[row["eppo"]][taxonomy] if row["eppo"] else ""
                # A weed whose group is also a crop's code (crop labelled at genus level,
                # e.g. poppy) keeps its own code.
                keep = not row["eppo"] or row["role"] == "crop" or group in crops
                row[taxonomy] = row["eppo"] if keep else group
        elif taxonomy not in first:
            raise KeyError(f"Taxonomy column {taxonomy!r} not in {path}")
        self.taxonomy = taxonomy
        self.taxa = taxa

    def keep_common_species(self, train_image_labels: Iterable[set[str]],
                            min_images: int) -> list[str]:
        """Switch the `species` taxonomy to species with at least `min_images` training images.

        1. A species code with at least `min_images` training images is a class.
        2. A rarer species goes to OTHER_CATEGORY.
        3. A coarser code (e.g. a genus-only label) goes to OTHER_CATEGORY, unless a class
           belongs to it: then it becomes an ignore region, since such a plant may well be
           of that class.
        4. Labels without an EPPO code (unidentified plants) stay ignore regions.

        `train_image_labels` holds the source labels of each training image. Returns report
        lines: every class, every species sent to OTHER_CATEGORY and every rule-3 decision.
        """
        if self.taxonomy != "species":
            raise ValueError(f"min_train_images needs taxonomy 'species', not {self.taxonomy!r}")
        if self.taxa is None:
            raise ValueError("min_train_images needs the EPPO taxa table")
        codes = {row["eppo"] for row in self.rows.values() if row["eppo"]}
        missing = sorted(code for code in codes if not self.taxa.get(code, {}).get("rank"))
        if missing:
            raise ValueError(f"EPPO codes missing from the taxa table: {', '.join(missing)}")

        images = Counter()  # training images per EPPO code
        for labels in train_image_labels:
            images.update({self.rows[label]["eppo"] for label in labels} - {""})
        classes = {code for code in codes
                   if self.taxa[code]["rank"] == "species" and images[code] >= min_images}
        ancestors = {self.taxa[code][rank] for code in classes for rank in PARENT_RANKS}

        decisions = {}
        for code in codes:
            if code in classes:
                decisions[code] = code
            elif self.taxa[code]["rank"] == "species" or code not in ancestors:
                decisions[code] = OTHER_CATEGORY
            else:
                decisions[code] = ""
        name = f"species_min{min_images}img"
        for row in self.rows.values():
            row[name] = decisions[row["eppo"]] if row["eppo"] else ""
        self.taxonomy = name

        def line(code: str) -> str:
            return f"  {code:6} {self.taxa[code]['name'][:30]:30} {images[code]:5} images"

        coarse = sorted(code for code in codes if self.taxa[code]["rank"] != "species")
        return [
            f"{len(classes)} classes (species with >= {min_images} training images):",
            *(line(code) for code in sorted(classes, key=lambda c: -images[c])),
            f"{OTHER_CATEGORY} (rarer species):",
            *(line(code) for code in sorted(codes - classes, key=lambda c: -images[c])
              if self.taxa[code]["rank"] == "species"),
            "Labels coarser than species:",
            *(f"{line(code)} -> {decisions[code] or 'ignore (one of its species is a class)'}"
              for code in coarse),
        ]

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
                "supercategory": "other" if name == OTHER_CATEGORY else _role(rows),
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
                      fractions: dict[str, float], seed: int,
                      labels_of: Callable[[T], Iterable[str]] | None = None) -> dict[str, list[T]]:
    """Assign whole groups to splits so each split gets about its fraction of the items and,
    with `labels_of`, of the items of each label.

    Balancing follows iterative stratification: groups holding the rarest labels are placed
    first, each in the split least filled (relative to its share) with the group's rarest
    label plus with items. A label found in fewer groups than there are splits can't reach
    every split.
    """
    by_group: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_group[group_of(item)].append(item)
    group_labels = {group: Counter(label for item in members
                                   for label in (labels_of(item) if labels_of else ()))
                    for group, members in by_group.items()}
    label_totals = sum(group_labels.values(), Counter())

    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    # Rarest labels first; the stable sort keeps the shuffled order among equals.
    groups.sort(key=lambda g: min((label_totals[label] for label in group_labels[g]),
                                  default=len(items) + 1))

    total = sum(fractions.values())
    shares = {name: fraction / total for name, fraction in fractions.items()}
    splits: dict[str, list[T]] = {name: [] for name in fractions}
    label_counts = {name: Counter() for name in fractions}
    for group in groups:
        labels = group_labels[group]
        rarest = min(labels, key=lambda label: (label_totals[label], label)) if labels else None

        def fill(name: str) -> float:
            # How full a split is relative to its target, for the rarest label plus for all
            # items: a split still without the label usually wins, however small its share,
            # while the item totals stay close to the fractions.
            items_fill = len(splits[name]) / (shares[name] * len(items))
            if rarest is None:
                return items_fill
            return label_counts[name][rarest] / (shares[name] * label_totals[rarest]) + items_fill

        name = min(splits, key=fill)
        splits[name].extend(by_group[group])
        label_counts[name].update(labels)
    return splits


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Compressed COCO RLE of a bool mask, JSON-serialisable."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray | None:
    """Bool mask of an annotation's segmentation (RLE or polygons), or None without one."""
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return None
    if isinstance(segmentation, list):  # polygons
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):  # uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


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


def is_small(box_width: float, box_height: float, mm_per_px: float,
             min_size_mm: float | None) -> bool:
    """Whether a plant is under the minimum size: its box's longer side, in mm."""
    return min_size_mm is not None and max(box_width, box_height) * mm_per_px < min_size_mm


def ignore_small_plants(coco: dict[str, Any], min_size_mm: float | None) -> dict[str, Any]:
    """Copy of `coco` where plants under `min_size_mm` (see `is_small`) become ignore regions
    (the `ignore` category, `iscrowd = 1`): neither learned nor scored, for any class. The
    size in mm uses each image's `mm_per_px`. None returns `coco` unchanged."""
    if min_size_mm is None:
        return coco
    ignore_id = next(c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY)
    mm_per_px = {i["id"]: i.get("mm_per_px") for i in coco["images"]}
    missing = [i["file_name"] for i in coco["images"] if i.get("mm_per_px") is None]
    if missing:
        raise ValueError(f"{len(missing)} images have no mm_per_px (e.g. {missing[0]}), "
                         "so plant sizes in mm are unknown")
    annotations = []
    for annotation in coco["annotations"]:
        width, height = annotation["bbox"][2:]
        if (not annotation.get("iscrowd")
                and is_small(width, height, mm_per_px[annotation["image_id"]], min_size_mm)):
            annotation = {**annotation, "category_id": ignore_id, "iscrowd": 1,
                          "ignored_category_id": annotation["category_id"]}
        annotations.append(annotation)
    return {**coco, "annotations": annotations}


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
