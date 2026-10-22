"""Reader for OPPD, the Open Plant Phenotyping Database (Aarhus University): weeds grown in
trays in Denmark, 2017-2019, photographed from above on many days, with a box per plant.

Raw layout of Dataset Ninja's copy "OPPD: Full Images" (Supervisely format, see
docs/datasets/oppd.md):
    meta.json                  the box classes (EPPO codes)
    ds0/img/<name>.jpg         4096x3000 RGB, one tray, rectified (black around it)
    ds0/ann/<name>.jpg.json    image tags (eppo = the species sown in the tray, date,
                               trial_id, box_id, ...) and one rectangle per plant, its class
                               an EPPO code

<name> = <trial>_<tray>_<time>, e.g. T01_Box053_2017-05-19T13-38-59-139. Each tray holds
one sown species and its images show the same plants growing, so the group is the tray
(trial and box: box ids repeat across trials). Image names are the file stems.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from weedseg.data.coco import ImageRecord, Instance

ANNOTATION_SUFFIX = ".jpg.json"


def image_names(raw_dir: Path) -> list[str]:
    return sorted(path.name.removesuffix(ANNOTATION_SUFFIX)
                  for path in (raw_dir / "ds0" / "ann").glob(f"*{ANNOTATION_SUFFIX}"))


def tray_of(name: str) -> str:
    """The tray an image shows (its group), e.g. "T01_Box053"."""
    trial, box, _ = name.split("_", 2)
    return f"{trial}_{box}"


def _annotation(raw_dir: Path, name: str) -> dict[str, Any]:
    path = raw_dir / "ds0" / "ann" / f"{name}{ANNOTATION_SUFFIX}"
    return json.loads(path.read_text(encoding="utf-8"))


def _boxes(annotation: dict[str, Any]) -> list[tuple[str, tuple[int, int, int, int]]]:
    """(source label, x0 y0 x1 y1) of each plant. Supervisely's rectangle corners are both
    inside the box, hence the + 1. Rectangles without extent are not plants: the download
    has two, both [[0, 0], [0, 0]]."""
    boxes = []
    for obj in annotation["objects"]:
        (left, top), (right, bottom) = obj["points"]["exterior"]
        if right > left and bottom > top:
            boxes.append((obj["classTitle"], (left, top, right + 1, bottom + 1)))
    return boxes


def box_labels(raw_dir: Path, name: str) -> list[tuple[str, int, int]]:
    """(source label, width, height) of each plant of an image, in px."""
    return [(label, x1 - x0, y1 - y0)
            for label, (x0, y0, x1, y1) in _boxes(_annotation(raw_dir, name))]


def iter_records(raw_dir: Path, names: list[str],
                 mm_per_px: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as source labels (the box classes, EPPO
    codes). Boxes only, no masks. `mm_per_px` is the dataset page's ground resolution."""
    for name in names:
        annotation = _annotation(raw_dir, name)
        tags = {tag["name"]: tag["value"] for tag in annotation["tags"]}
        instances = [Instance(source_label=label, bbox_xyxy=box)
                     for label, box in _boxes(annotation)]
        attributes = {"trial": tags["trial_id"], "tray": tags["box_id"],
                      "date": tags["date"][:10], "sown_species": tags["eppo"]}
        if mm_per_px is not None:
            attributes["mm_per_px"] = mm_per_px
            attributes["mm_per_px_estimated"] = False
        yield ImageRecord(file_name=f"{name}.jpg", width=annotation["size"]["width"],
                          height=annotation["size"]["height"], group=tray_of(name),
                          instances=instances, attributes=attributes)
