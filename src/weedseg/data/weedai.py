"""Reader for datasets downloaded from Weed-AI (https://weed-ai.sydney.edu.au), the University
of Sydney's repository of weed images, in its WeedCOCO format (see docs/datasets/weedai.md).

Raw layout: one folder per Weed-AI upload (a "part"; a dataset may be split over several,
e.g. one per crop), as unpacked from its download:
    <part>/weedcoco.json   COCO plus WeedCOCO fields: categories "<role>: <latin name>"
                           (e.g. "crop: beta vulgaris", "weed: UNSPECIFIED"), boxes, and
                           agcontexts (camera, height, location, growth stage)
    <part>/images/<file>   the images; file_name in the JSON may or may not carry "images/"

Image names are "<part>/<file name without folder>". The group comes from the original
file name (before Roboflow's ".rf.<hash>" and its "_jpg" ending), through a dataset's
`group_pattern`; uploads that lost the original names can't be split by group.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

from weedseg.data.coco import ImageRecord, Instance

ANNOTATIONS = "weedcoco.json"
# Roboflow exports name a file <original stem>_<ext>.rf.<hash>.<ext>.
ROBOFLOW = re.compile(r"(.+?)(?:_(?:jpe?g|png|JPE?G|PNG))?\.rf\.[0-9a-f]+\.\w+")


@dataclass(frozen=True)
class Grouping:
    """How an image's group comes from its original name: the first capture group of
    `pattern` (matched from the start). With `block`, that capture is a number and the group
    is its block of `block` consecutive numbers, e.g. "1000-1019" for 1014 with blocks of 20."""

    pattern: str
    block: int | None = None

    def group_of(self, source_name: str) -> str:
        match = re.match(self.pattern, source_name)
        if match is None:
            raise ValueError(f"{source_name!r} doesn't match group_pattern {self.pattern!r}")
        if self.block is None:
            return match.group(1)
        start = int(match.group(1)) // self.block * self.block
        return f"{start}-{start + self.block - 1}"


@dataclass(frozen=True)
class Scale:
    """An image's mm/px: `mm_per_px` for all images, or `footprint_mm`, the ground length
    of the image's longer side, for a dataset whose images were resized to different
    sizes. Both are estimates."""

    mm_per_px: float | None = None
    footprint_mm: float | None = None

    def __post_init__(self):
        if (self.mm_per_px is None) == (self.footprint_mm is None):
            raise ValueError("Give exactly one of mm_per_px_estimate and footprint_mm_estimate")

    def of(self, width: int, height: int) -> float:
        return self.mm_per_px or self.footprint_mm / max(width, height)


def source_name(file_name: str) -> str:
    """The original file stem of a Weed-AI image: Roboflow's ending removed, e.g.
    "bbro_bbro_07_05_2021_v_0_0" for "bbro_bbro_07_05_2021_v_0_0_png.rf.2b16[...].jpg"."""
    name = PurePosixPath(file_name).name
    match = ROBOFLOW.fullmatch(name)
    return match.group(1) if match else PurePosixPath(name).stem


@cache
def _load(path: Path) -> dict[str, Any]:
    coco = json.loads(path.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {}
    for image in coco["images"]:
        images[image["id"]] = {**image, "boxes": []}
    for annotation in coco["annotations"]:
        x, y, w, h = annotation["bbox"]
        images[annotation["image_id"]]["boxes"].append(
            (categories[annotation["category_id"]], (x, y, x + w, y + h)))
    return {PurePosixPath(i["file_name"]).name: i for i in images.values()}


def _image(raw_dir: Path, name: str) -> dict[str, Any]:
    part, file = name.split("/", 1)
    return _load(raw_dir / part / ANNOTATIONS)[file]


def image_names(raw_dir: Path, parts: list[str]) -> list[str]:
    """The images of the given parts, "<part>/<file name>"."""
    names = []
    for part in parts:
        path = raw_dir / part / ANNOTATIONS
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/download_datasets.py")
        names.extend(f"{part}/{file}" for file in _load(path))
    return sorted(names)


def duplicates(raw_dir: Path, names: list[str]) -> list[str]:
    """The images whose file is byte for byte that of an earlier name (in `names` order), e.g.
    ImageWeeds' photos uploaded twice under two numbers and labelled twice."""
    first, repeated = {}, []
    for name in names:
        part, file = name.split("/", 1)
        digest = hashlib.sha1((raw_dir / part / "images" / file).read_bytes()).hexdigest()
        if digest in first:
            repeated.append(name)
        else:
            first[digest] = name
    return repeated


def _boxes(raw_dir: Path, name: str, drop: frozenset[str]) -> list[tuple[str, tuple]]:
    return [(label, box) for label, box in _image(raw_dir, name)["boxes"] if label not in drop]


def box_labels(raw_dir: Path, name: str,
               drop: frozenset[str] = frozenset()) -> list[tuple[str, float, float]]:
    """(source label, width, height) of each plant of an image, in px; labels in `drop`
    (boxes that aren't plants, e.g. VCD's stems) left out."""
    return [(label, x1 - x0, y1 - y0) for label, (x0, y0, x1, y1) in _boxes(raw_dir, name, drop)]


def mm_per_px(raw_dir: Path, name: str, scale: Scale) -> float:
    image = _image(raw_dir, name)
    return scale.of(image["width"], image["height"])


def iter_records(raw_dir: Path, names: list[str], grouping: Grouping, scale: Scale,
                 drop: frozenset[str] = frozenset()) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as source labels (the WeedCOCO category
    names). Boxes only, no masks. Image fields: `part`, `source_name` (the original file
    stem), `mm_per_px` (an estimate, see `Scale`)."""
    for name in names:
        part, file = name.split("/", 1)
        image = _image(raw_dir, name)
        original = source_name(file)
        yield ImageRecord(
            file_name=f"{part}/images/{file}", width=image["width"], height=image["height"],
            group=grouping.group_of(original),
            instances=[Instance(source_label=label, bbox_xyxy=box)
                       for label, box in _boxes(raw_dir, name, drop)],
            attributes={"part": part, "source_name": original,
                        "mm_per_px": scale.of(image["width"], image["height"]),
                        "mm_per_px_estimated": True})
