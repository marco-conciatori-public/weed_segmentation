"""Reader for the CornWeed dataset (DFKI and Hochschule Osnabrück, FedCSIS 2023): maize rows in
one field ("Langsenkamp"), filmed straight down with an Intel RealSense D435i on a tractor hoe
(1280x720) and on the BoniRob research platform (640x480), July-August 2021. The scale is
estimated per image width (docs/datasets/cornweed.md).

Raw layout after download (see docs/datasets/cornweed.md):
    maize_dataset/coco_annotations/{train,valid}_2022.json   COCO boxes, classes Weeds, Maize
    maize_dataset/data/<run>_<frame>.png                      the labelled images
    maize_dataset/testing/                                   unlabelled images (not used)

<run> is "Langsenkamp_<yymmdd>_color" (tractor) or "Langsenkamp_<yymmdd>_bonirob_color"; the
frame number counts the run's video frames, so neighbouring numbers show the same plants.
Image names are the file names.
"""

import json
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from weedseg.data.coco import ImageRecord, Instance

ROOT = "maize_dataset"
ANNOTATIONS = ("coco_annotations/train_2022.json", "coco_annotations/valid_2022.json")
NAME = re.compile(r"(?P<run>(?P<field>[A-Za-z]+)_(?P<date>\d{6})_(?P<platform>bonirob_)?color)"
                  r"_(?P<frame>\d+)\.png")


@cache
def _load(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """{file name: image with its `boxes` [(label, (x0, y0, x1, y1))]}, over both JSON files."""
    images = {}
    for relative in ANNOTATIONS:
        path = raw_dir / ROOT / relative
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/download_datasets.py")
        coco = json.loads(path.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        by_id = {}
        for image in coco["images"]:
            if image["file_name"] in images:
                raise ValueError(f"{image['file_name']} is in both annotation files")
            by_id[image["id"]] = images[image["file_name"]] = {**image, "boxes": []}
        for annotation in coco["annotations"]:
            x, y, w, h = annotation["bbox"]
            by_id[annotation["image_id"]]["boxes"].append(
                (categories[annotation["category_id"]], (x, y, x + w, y + h)))
    return images


def parse(name: str) -> dict[str, Any]:
    """Run, field, date (YYYY-MM-DD), platform and frame number of an image name."""
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected CornWeed image name {name!r}")
    date = match["date"]
    return {"run": match["run"], "field": match["field"],
            "date": f"20{date[:2]}-{date[2:4]}-{date[4:]}",
            "platform": "bonirob" if match["platform"] else "tractor",
            "frame": int(match["frame"])}


def group_of(name: str, block: int) -> str:
    """The run and the block of `block` consecutive frames an image belongs to, e.g.
    "Langsenkamp_210727_color_00750" for frame 773 with blocks of 50."""
    info = parse(name)
    return f"{info['run']}_{info['frame'] // block * block:05d}"


def image_names(raw_dir: Path) -> list[str]:
    return sorted(_load(raw_dir))


def thin_frames(names: list[str], stride: int) -> list[str]:
    """The frames whose number is a multiple of `stride`: one in every `stride` frames of a
    run. Consecutive frames move about half an image, so 2 shows the ground about once."""
    if stride < 1:
        raise ValueError(f"frame_stride must be at least 1, not {stride}")
    return [name for name in names if parse(name)["frame"] % stride == 0]


def mm_per_px(raw_dir: Path, name: str, by_width: dict[int, float]) -> float:
    """The image's mm/px, an estimate per image width (that is, per camera setting)."""
    width = _load(raw_dir)[name]["width"]
    if width not in by_width:
        raise ValueError(f"No mm/px for images {width} px wide (mm_per_px_by_width)")
    return by_width[width]


def box_labels(raw_dir: Path, name: str) -> list[tuple[str, float, float]]:
    """(source label, width, height) of each plant of an image, in px."""
    return [(label, x1 - x0, y1 - y0) for label, (x0, y0, x1, y1) in _load(raw_dir)[name]["boxes"]]


def iter_records(raw_dir: Path, names: list[str], block: int,
                 by_width: dict[int, float]) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as the dataset's class names (Weeds,
    Maize). Image fields: `run`, `date`, `platform`, `frame`, `mm_per_px` (estimated)."""
    for name in names:
        image = _load(raw_dir)[name]
        info = parse(name)
        yield ImageRecord(
            file_name=name, width=image["width"], height=image["height"],
            group=group_of(name, block),
            instances=[Instance(source_label=label, bbox_xyxy=box) for label, box in image["boxes"]],
            attributes={"run": info["run"], "date": info["date"], "platform": info["platform"],
                        "frame": info["frame"], "mm_per_px": mm_per_px(raw_dir, name, by_width),
                        "mm_per_px_estimated": True})
