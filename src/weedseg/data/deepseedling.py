"""Reader for DeepSeedling (Jiang, Li, Paterson and Robertson, University of Georgia, Plant
Methods 2019): cotton seedlings filmed from about 0.5 m in Corpus Christi, Texas (2015) and
Watkinsville, Georgia (2015, 2018), video frames labelled in VoTT 1.5 with a box for every
seedling (`Plant`) and every weed (`Weed`, no species) over about 30 px on the labelling
screen. The scale is estimated per set (docs/datasets/deepseedling.md).

Raw layout after download (configs/deepseedling.yaml unpacks each zip into its own folder):
    <set>/<folder>/<frame>.jpg    the frames of one or more videos, e.g. UGA2015/102/1001.jpg
    <set>/<folder>.json           their VoTT 1.5 labels
<set> is TAMU2015, UGA2015 or UGA2018. TAMU2015 also has Row10.mp4 with Row10.mp4.json,
labelled on the video file: not used (it has no folder of frames).

VoTT 1.5 labels: {"frames": {"<i>": [box, ...]}, ...}. Frame i is the i-th image of the
folder in the order Windows lists it (by name, ignoring case: "1001.jpg" before "121.jpg");
each box has x1, y1, x2, y2 in the coordinates of the screen VoTT drew it on (the box's
`width` and `height`), and `tags` ["Plant"] or ["Weed"]. Frames missing from "frames" were not
labelled, and a frame listed without boxes may have been skipped, so neither is used.

The frame number ends the file name ("Frame05461", "1001", "ENGR_1_000081"); what comes
before it names the video ("ENGR_1_": UGA2018 keeps six videos in one folder). A video is
<set>/<folder>/<prefix>, and its frames in number order are its time order.
Image names are paths from the raw folder, e.g. "UGA2015/102/1001.jpg".
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image

from weedseg.data.coco import ImageRecord, Instance

IMAGE_SUFFIXES = (".jpg", ".jpeg")
FRAME = re.compile(r"(?P<prefix>.*?)(?P<number>\d+)", re.IGNORECASE)
LABELS = ("Plant", "Weed")


def parse(name: str) -> dict[str, Any]:
    """Set, folder, video and frame number of an image name such as
    "UGA2018/Images/ENGR_1_000081.jpg" (video "UGA2018/Images/ENGR_1")."""
    parts = name.split("/")
    if len(parts) != 3:
        raise ValueError(f"Unexpected DeepSeedling image name {name!r}")
    set_name, folder, file_name = parts
    match = FRAME.fullmatch(Path(file_name).stem)
    if match is None:
        raise ValueError(f"No frame number in DeepSeedling image name {name!r}")
    video = "/".join(filter(None, (set_name, folder, match["prefix"].rstrip("_"))))
    return {"set": set_name, "folder": folder, "video": video, "frame": int(match["number"])}


def _vott_order(names: list[str]) -> list[str]:
    """File names in the order VoTT numbered them: Windows' (by name, ignoring case)."""
    return sorted(names, key=str.upper)


@cache
def _load(raw_dir: Path, sets: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """{image name: image with `width`, `height`, `boxes` [(label, (x0, y0, x1, y1))] in
    image px, `video` and `position` (its place among the video's labelled frames, in time
    order)}, over the folders of `sets` that have a VoTT file; only frames with boxes."""
    images = {}
    for set_name in sets:
        set_dir = raw_dir / set_name
        if not set_dir.is_dir():
            raise FileNotFoundError(f"{set_dir} missing; run scripts/download_datasets.py")
        for label_file in sorted(set_dir.glob("*.json")):
            folder = set_dir / label_file.stem
            if not folder.is_dir():  # e.g. Row10.mp4.json, labelled on the video
                continue
            files = _vott_order([p.name for p in folder.iterdir()
                                 if p.suffix.lower() in IMAGE_SUFFIXES])
            vott = json.loads(label_file.read_text(encoding="utf-8"))
            for index, boxes in vott["frames"].items():
                if not boxes:
                    continue
                index = int(index)
                if index >= len(files):
                    raise ValueError(f"{label_file}: frame {index}, but {folder} has "
                                     f"{len(files)} images")
                name = f"{set_name}/{folder.name}/{files[index]}"
                with Image.open(raw_dir / name) as image:
                    width, height = image.size
                clipped = [_box(b, width, height, name) for b in boxes]
                clipped = [(label, box) for label, box in clipped  # drop boxes off the image
                           if box[2] > box[0] and box[3] > box[1]]
                if clipped:
                    images[name] = {"width": width, "height": height, "boxes": clipped}
    # time order within each video, for thinning and grouping
    videos = defaultdict(list)
    for name in images:
        info = parse(name)
        videos[info["video"]].append((info["frame"], name))
    for video, frames in videos.items():
        for position, (_, name) in enumerate(sorted(frames)):
            images[name].update(video=video, position=position)
    return images


def _box(box: dict[str, Any], width: int, height: int,
         name: str) -> tuple[str, tuple[float, float, float, float]]:
    """A VoTT box as (label, (x0, y0, x1, y1)) in image px, clipped to the image."""
    tags = sorted(set(box["tags"]))
    if not tags or any(tag not in LABELS for tag in tags):
        raise ValueError(f"{name}: box with tags {tags}, expected one of {LABELS}")
    # a few boxes carry both tags: a plant of unknown species, label "Plant+Weed"
    sx, sy = width / box["width"], height / box["height"]
    x0, x1 = sorted((box["x1"] * sx, box["x2"] * sx))
    y0, y1 = sorted((box["y1"] * sy, box["y2"] * sy))
    return "+".join(tags), (max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1))


def image_names(raw_dir: Path, sets: dict[str, dict[str, Any]]) -> list[str]:
    """The labelled frames (with at least one box) of `sets` (config `convert.sets`)."""
    return sorted(_load(raw_dir, tuple(sorted(sets))))


def thin_frames(raw_dir: Path, names: list[str], sets: dict[str, dict[str, Any]]) -> list[str]:
    """One labelled frame in every `frame_stride` (per set) of each video, in time order:
    neighbouring frames show the same plants."""
    images = _load(raw_dir, tuple(sorted(sets)))
    kept = []
    for name in names:
        stride = sets[parse(name)["set"]]["frame_stride"]
        if stride < 1:
            raise ValueError(f"frame_stride must be at least 1, not {stride}")
        if images[name]["position"] % stride == 0:
            kept.append(name)
    return kept


def group_of(raw_dir: Path, name: str, sets: dict[str, dict[str, Any]]) -> str:
    """The video and the block of `group_block` (per set) consecutive labelled frames an image
    belongs to, e.g. "UGA2015/102_0060" for the 61st-120th labelled frames of video 102."""
    image = _load(raw_dir, tuple(sorted(sets)))[name]
    block = sets[parse(name)["set"]]["group_block"]
    return f"{image['video']}_{image['position'] // block * block:04d}"


def mm_per_px(name: str, sets: dict[str, dict[str, Any]]) -> float:
    """The image's mm/px: its set's estimate."""
    return sets[parse(name)["set"]]["mm_per_px"]


def box_labels(raw_dir: Path, name: str,
               sets: dict[str, dict[str, Any]]) -> list[tuple[str, float, float]]:
    """(source label, width, height) of each plant of an image, in px."""
    return [(label, x1 - x0, y1 - y0)
            for label, (x0, y0, x1, y1) in _load(raw_dir, tuple(sorted(sets)))[name]["boxes"]]


def iter_records(raw_dir: Path, names: list[str],
                 sets: dict[str, dict[str, Any]]) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as the dataset's tags (Plant, Weed).
    Image fields: `set`, `video`, `frame`, `mm_per_px` (estimated)."""
    images = _load(raw_dir, tuple(sorted(sets)))
    for name in names:
        image, info = images[name], parse(name)
        yield ImageRecord(
            file_name=name, width=image["width"], height=image["height"],
            group=group_of(raw_dir, name, sets),
            instances=[Instance(source_label=label, bbox_xyxy=box) for label, box in image["boxes"]],
            attributes={"set": info["set"], "video": info["video"], "frame": info["frame"],
                        "mm_per_px": mm_per_px(name, sets), "mm_per_px_estimated": True})
