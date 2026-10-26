"""Reader for GROUNDBASED_WEED (CSIC; Moreno, Rivera and Andújar, Data in Brief 2025): hand-held
top-down photos of commercial tomato fields in Santa Amalia (Badajoz, Spain), May-June 2021
and 2022, with a box per plant.

Raw layout (see docs/datasets/groundbased_weed.md):
    TOMATO_<n>/Images/<stem>.JPG       TOMATO_1 = 2021, TOMATO_2 = 2022. Photos IMG_<number>
                                       5184x3888 or turned 3888x5184; TOMATO_1 also has cut-outs
                                       of quadrat photos, <quadrat>_IMG_<number>, ~3500 px
    TOMATO_<n>/Annotations/<stem>.xml  Pascal VOC: one box per plant, its class an EPPO code
                                       or NR (not recognised)
    Annotations_extra/                 the same boxes as COCO and YOLO files; not used

Image names are "TOMATO_<n>/<stem>". The images carry no field or date (TOMATO_1 has no EXIF
at all), so the group is a block of consecutive camera numbers: the photos of one stretch of
a walk through a field.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from weedseg.data.coco import ImageRecord, Instance

PARTS = {"TOMATO_1": 2021, "TOMATO_2": 2022}  # the year of each part, from the paper
# IMG_4130, 02_IMG_4550 (quadrat 2); one file lacks the underscore: 25IMG_4626.
NAME = re.compile(r"(?:(\d+)_?)?IMG_(\d+)")


def image_names(raw_dir: Path) -> list[str]:
    """The annotated images, "<part>/<file stem>"."""
    return sorted(f"{part}/{path.stem}" for part in PARTS
                  for path in (raw_dir / part / "Annotations").glob("*.xml"))


def _parse(name: str) -> tuple[str, int | None, int]:
    """(part, quadrat or None, camera number) of an image name."""
    part, stem = name.split("/")
    match = NAME.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unexpected image name {name!r}")
    quadrat, number = match.groups()
    return part, None if quadrat is None else int(quadrat), int(number)


def group_of(name: str, block: int) -> str:
    """The block of `block` consecutive camera numbers an image belongs to (its group), e.g.
    "TOMATO_1_4120" for TOMATO_1/IMG_4130 with blocks of 20."""
    part, _, number = _parse(name)
    return f"{part}_{number // block * block}"


def _annotation(raw_dir: Path, name: str) -> ET.Element:
    part, stem = name.split("/")
    return ET.parse(raw_dir / part / "Annotations" / f"{stem}.xml").getroot()


def _boxes(annotation: ET.Element) -> list[tuple[str, tuple[float, ...], dict[str, bool]]]:
    """(source label, x0 y0 x1 y1, flags) of each plant; flags hold VOC's `difficult` and
    `truncated` when set."""
    boxes = []
    for obj in annotation.iter("object"):
        box = obj.find("bndbox")
        flags = {flag: True for flag in ("difficult", "truncated")
                 if (obj.findtext(flag) or "0").strip() == "1"}
        boxes.append((obj.findtext("name").strip(),
                      tuple(float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")),
                      flags))
    return boxes


def box_labels(raw_dir: Path, name: str) -> list[tuple[str, float, float]]:
    """(source label, width, height) of each plant of an image, in px."""
    return [(label, x1 - x0, y1 - y0)
            for label, (x0, y0, x1, y1), _ in _boxes(_annotation(raw_dir, name))]


def iter_records(raw_dir: Path, names: list[str], block: int,
                 mm_per_px_estimate: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as source labels (EPPO codes and NR).
    Boxes only, no masks. `block` sets the groups (see `group_of`)."""
    for name in names:
        part, quadrat, _ = _parse(name)
        annotation = _annotation(raw_dir, name)
        size = annotation.find("size")
        instances = [Instance(source_label=label, bbox_xyxy=box, attributes=flags)
                     for label, box, flags in _boxes(annotation)]
        attributes = {"subset": part, "year": PARTS[part]}
        if quadrat is not None:
            attributes["quadrat"] = quadrat
        if mm_per_px_estimate is not None:
            attributes["mm_per_px"] = mm_per_px_estimate
            attributes["mm_per_px_estimated"] = True
        yield ImageRecord(file_name=f"{part}/Images/{annotation.findtext('filename')}",
                          width=int(size.findtext("width")), height=int(size.findtext("height")),
                          group=group_of(name, block), instances=instances,
                          attributes=attributes)
