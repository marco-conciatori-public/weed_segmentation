"""Reader for PhenoBench (University of Bonn): drone images of a sugar beet field at Campus
Klein-Altendorf on 15 May, 26 May and 5 June 2020, with an instance mask per plant.

Raw layout (see docs/datasets/phenobench.md):
    PhenoBench/<train|val>/images/<MM-DD>_<index>_<photo>.png  1024x1024 RGB
    PhenoBench/<train|val>/semantics/<same name>        per pixel: 0 soil, 1 crop, 2 weed,
                                                        3 partial crop, 4 partial weed
    PhenoBench/<train|val>/plant_instances/<same name>  plant ids, 0 = no plant
    PhenoBench/test/images/                             no labels, not used

The images are overlapping windows of the drone's 11,664 x 8,750 px photos, so a plant can
appear in several images. "Partial" plants are less than half visible (cut by the image
border or covered by a neighbour). Image names are "<split>/<file stem>"; the group is the
drone photo.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from weedseg.data.coco import ImageRecord, Instance

LABELLED_SPLITS = ("train", "val")
SOURCE_LABELS = {1: "crop", 2: "weed", 3: "partial_crop", 4: "partial_weed"}
PARTIAL_LABELS = {"partial_crop", "partial_weed"}
YEAR = 2020


def image_names(raw_dir: Path) -> list[str]:
    """The labelled images, "<split>/<file stem>"."""
    return sorted(f"{split}/{path.stem}" for split in LABELLED_SPLITS
                  for path in (raw_dir / split / "semantics").glob("*.png"))


def photo_of(name: str) -> str:
    """The drone photo an image was cut from (its group), e.g. "05-15_P0030852"."""
    date, _, photo = name.split("/", 1)[1].split("_")
    return f"{date}_{photo}"


def own_splits(names: list[str]) -> dict[str, list[str]]:
    """PhenoBench's own split, which separates crop rows: its train and val images as they
    are. Its test images have no public labels, so the test split is empty."""
    return {"train": [name for name in names if name.startswith("train/")],
            "val": [name for name in names if name.startswith("val/")],
            "test": []}


def _plants(raw_dir: Path, name: str) -> tuple[list[tuple[str, np.ndarray]], tuple[int, int]]:
    """(source label, bool mask) of each plant of an image, and the image's (height, width)."""
    split, stem = name.split("/", 1)
    paths = [raw_dir / split / folder / f"{stem}.png" for folder in ("semantics",
                                                                    "plant_instances")]
    semantic, instances = (cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in paths)
    for path, image in zip(paths, (semantic, instances)):
        if image is None:
            raise FileNotFoundError(path)
    plants = []
    for plant_id in np.unique(instances):
        if plant_id == 0:
            continue
        mask = instances == plant_id
        labels, counts = np.unique(semantic[mask], return_counts=True)
        label = int(labels[counts.argmax()])
        if label not in SOURCE_LABELS:
            raise ValueError(f"{name}: plant {plant_id} has the semantic label {label}")
        plants.append((SOURCE_LABELS[label], mask))
    return plants, semantic.shape


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return x, y, x + w, y + h


def box_labels(raw_dir: Path, name: str) -> list[tuple[str, int, int]]:
    """(source label, width, height) of each plant of an image, in px."""
    plants, _ = _plants(raw_dir, name)
    return [(label, x1 - x0, y1 - y0)
            for label, (x0, y0, x1, y1) in ((label, _box(mask)) for label, mask in plants)]


def iter_records(raw_dir: Path, names: list[str], mm_per_px: float | None = None,
                 ignore_partial: bool = True) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as source labels (crop, weed,
    partial_crop, partial_weed). With `ignore_partial`, plants less than half visible
    become ignore regions. `mm_per_px` is the paper's ground resolution."""
    for name in names:
        plants, (height, width) = _plants(raw_dir, name)
        instances = [Instance(source_label=label, bbox_xyxy=_box(mask), mask=mask,
                              ignore=ignore_partial and label in PARTIAL_LABELS)
                     for label, mask in plants]
        split, stem = name.split("/", 1)
        date, _, photo = stem.split("_")
        attributes = {"date": f"{YEAR}-{date}", "photo": photo}
        if mm_per_px is not None:
            attributes["mm_per_px"] = mm_per_px
            attributes["mm_per_px_estimated"] = False
        yield ImageRecord(file_name=f"{split}/images/{stem}.png", width=width, height=height,
                          group=photo_of(name), instances=instances, attributes=attributes)
