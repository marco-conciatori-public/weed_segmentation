"""Reader for Sugar Beets 2016 (University of Bonn, IJRR 2017): one sugar beet field at Campus
Klein-Altendorf, photographed straight down from the BoniRob robot (JAI camera, ~85 cm,
1296x966); crop and weed masks per pixel, no plant IDs. Only the first part of the labelled
images' RAR archive is used (the day 2016-05-23), as an extra test set.

Raw layout after unpacking (see docs/datasets/sugarbeets2016.md):
    ijrr_sugarbeets_2016_annotations/CKA_<yymmdd>/images/rgb/<run>_frame<n>.png
    ijrr_sugarbeets_2016_annotations/CKA_<yymmdd>/annotations/dlp/iMapCleaned/<same>.png
        index map: 0 soil, 10000 sugar beet, 2 and 20100 weed
    ijrr_sugarbeets_2016_annotations/CKA_<yymmdd>/annotations/dlp/colorCleaned/  (the same,
        in colour; not used)

<run> is "bonirob_<yyyy-mm-dd>-<hh-mm-ss>_<n>", one recording; frames of a run are
consecutive. Image names are paths under ijrr_sugarbeets_2016_annotations/, e.g.
"CKA_160523/images/rgb/bonirob_2016-05-23-10-47-22_2_frame200.png".

Plants are the masks' connected patches (8-connected), one box each: touching plants become
one box and a plant whose leaves don't touch in the image becomes several.
"""

import re
import subprocess
from collections.abc import Iterator
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2
import numpy as np

from weedseg.data.coco import ImageRecord, Instance

ROOT = "ijrr_sugarbeets_2016_annotations"
RGB, LABELS = "images/rgb", "annotations/dlp/iMapCleaned"
NAME = re.compile(r"(?P<run>bonirob_(?P<date>\d{4}-\d\d-\d\d)-\d\d-\d\d-\d\d_\d+)"
                  r"_frame(?P<frame>\d+)\.png")

Classes = tuple[tuple[str, tuple[int, ...]], ...]  # (source label, index map values)


def unpack(raw_dir: Path, archive: str, seven_zip: str) -> None:
    """Unpack a RAR part into `raw_dir` with 7-Zip, once (a marker file records it). A part on
    its own ends inside a file of the next part: files that come out shorter than the
    archive lists them are removed."""
    marker = raw_dir / f".unpacked_{archive}"
    if marker.exists():
        return
    path = raw_dir / archive
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/download_datasets.py")
    # 7-Zip reports an error for a part on its own (the next part is missing): its exit code
    # is not checked, the files are.
    listing = subprocess.run([seven_zip, "l", "-slt", str(path)], check=False,
                             capture_output=True, text=True, errors="replace").stdout
    sizes = {}
    for block in listing.split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" in fields and fields.get("Folder") == "-" and fields.get("Size", "").isdigit():
            sizes[PureWindowsPath(fields["Path"]).as_posix()] = int(fields["Size"])
    if not sizes:
        raise RuntimeError(f"7-Zip listed no files in {path}:\n{listing[-2000:]}")
    print(f"unpacking {archive} ({len(sizes)} files) with 7-Zip ...")
    subprocess.run([seven_zip, "x", "-y", f"-o{raw_dir}", str(path)], check=False,
                   capture_output=True)
    incomplete = [p for p, size in sizes.items()
                  if not (raw_dir / p).is_file() or (raw_dir / p).stat().st_size != size]
    for p in incomplete:
        (raw_dir / p).unlink(missing_ok=True)
    if len(incomplete) > 1:
        raise RuntimeError(f"{len(incomplete)} files of {archive} didn't unpack, e.g. "
                           f"{incomplete[0]}")
    marker.write_text(f"{len(sizes) - len(incomplete)} files\n", encoding="utf-8")


def parse(name: str) -> dict[str, Any]:
    """Run, date (YYYY-MM-DD) and frame number of an image name."""
    match = NAME.fullmatch(PureWindowsPath(name).name)
    if match is None:
        raise ValueError(f"Unexpected Sugar Beets 2016 image name {name!r}")
    return {"run": match["run"], "date": match["date"], "frame": int(match["frame"])}


def label_path(raw_dir: Path, name: str) -> Path:
    return raw_dir / ROOT / name.replace(RGB, LABELS)


def image_names(raw_dir: Path) -> list[str]:
    """Images that have an index map, as paths under ROOT."""
    root = raw_dir / ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"{root} missing; run scripts/convert_sugarbeets2016.py "
                                "(it unpacks the archive)")
    names = (p.relative_to(root).as_posix() for p in root.glob(f"*/{RGB}/*.png"))
    return sorted(n for n in names if label_path(raw_dir, n).exists())


def group_of(name: str) -> str:
    return parse(name)["run"]


@cache
def patches(raw_dir: Path, name: str, classes: Classes,
            min_px: int) -> tuple[tuple[int, int], tuple[tuple[str, tuple[int, ...], int], ...]]:
    """Image size and the patches of each class: (source label, (x0, y0, x1, y1), pixels)."""
    labels = cv2.imread(str(label_path(raw_dir, name)), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise OSError(f"Can't read {label_path(raw_dir, name)}")
    found = []
    for label, values in classes:
        mask = np.isin(labels, values).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for x, y, w, h, area in stats[1:count].tolist():
            if area >= min_px:
                found.append((label, (x, y, x + w, y + h), area))
    return (labels.shape[1], labels.shape[0]), tuple(found)


def box_labels(raw_dir: Path, name: str, classes: Classes,
               min_px: int) -> list[tuple[str, int, int]]:
    """(source label, width, height) of each patch of an image, in px."""
    return [(label, x1 - x0, y1 - y0)
            for label, (x0, y0, x1, y1), _ in patches(raw_dir, name, classes, min_px)[1]]


def iter_records(raw_dir: Path, names: list[str], classes: Classes, min_px: int,
                 mm_per_px: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels `beet` and `weed`, one per patch.
    Annotation field `patch_px`: the patch's pixels (a beet patch far bigger than the others
    is likely several plants). Image fields: `run`, `date`, `frame`, `mm_per_px`
    (estimated)."""
    for name in names:
        (width, height), found = patches(raw_dir, name, classes, min_px)
        attributes = dict(parse(name))
        if mm_per_px is not None:
            attributes["mm_per_px"] = mm_per_px
            attributes["mm_per_px_estimated"] = True
        yield ImageRecord(
            file_name=name, width=width, height=height, group=group_of(name),
            instances=[Instance(source_label=label, bbox_xyxy=box,
                                attributes={"patch_px": area})
                       for label, box, area in found],
            attributes=attributes)
