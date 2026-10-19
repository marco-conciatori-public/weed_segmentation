"""Reader for the ACRE Crop-Weed Dataset (INRAE Montoldre, France, June 2022).

Raw layout (see docs/datasets/acre.md):
    data/<Bean|Maize>/<batch>/rgb-<time>.jpg   2046x1080 RGB, top-down, rectified
    data/<Bean|Maize>/<batch>/rgb-<time>.xml   one <clipping> per plant: polygon, stem point
                                               ("center"), class (crop/weed/unknow), name

<batch> is the start of a recording, e.g. 2022-06-08-16-09: 100 consecutive images of one
plot, so neighbouring images may show the same plant. Image names are "<crop>/<batch>/<stem>".
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from weedseg.data.coco import ImageRecord, Instance

UNKNOWN_CLASS = "unknow"  # the annotators' class for plants they could not tell
UNNAMED = "unnamed"  # source label of a plant with an empty name


def image_names(raw_dir: Path) -> list[str]:
    return sorted(xml.relative_to(raw_dir / "data").with_suffix("").as_posix()
                  for xml in (raw_dir / "data").glob("*/*/*.xml"))


def batch_of(name: str) -> str:
    """Recording batch of an image name (the split group), e.g. "Bean/2022-06-08-16-09"."""
    return name.rsplit("/", 1)[0]


def source_label(clipping: ET.Element) -> str:
    """The plant's name, lower-cased; "unknow" when the annotators marked it as unknown."""
    if clipping.findtext("class") == UNKNOWN_CLASS:
        return UNKNOWN_CLASS
    return (clipping.findtext("name") or "").strip().lower() or UNNAMED


def _plants(raw_dir: Path, name: str) -> tuple[ET.Element, list[tuple[ET.Element, np.ndarray]]]:
    root = ET.parse(raw_dir / "data" / f"{name}.xml").getroot()
    plants = []
    for clipping in root.findall("data/clippings/clipping"):
        points = np.array([(int(p.get("x")), int(p.get("y")))
                           for p in clipping.findall("points/point")], np.int32).reshape(-1, 2)
        if len(points):
            plants.append((clipping, points))
    return root, plants


def box_labels(raw_dir: Path, name: str) -> list[tuple[str, int, int]]:
    """(source label, width, height) of each plant of an image, in px, from its polygon."""
    _, plants = _plants(raw_dir, name)
    return [(source_label(clipping), int(np.ptp(points[:, 0])) + 1, int(np.ptp(points[:, 1])) + 1)
            for clipping, points in plants]


def iter_records(raw_dir: Path, names: list[str],
                 mm_per_px_estimate: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as source labels (see `source_label`).

    Each polygon becomes the plant's mask. Polygons with fewer than 3 points, and plants the
    annotators did not trust, become ignore regions.
    """
    for name in names:
        root, plants = _plants(raw_dir, name)
        width = int(root.findtext("metadata/size/width"))
        height = int(root.findtext("metadata/size/height"))
        instances = []
        for clipping, points in plants:
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0) + 1
            mask = np.zeros((height, width), np.uint8)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
            else:  # too few points to outline: the box is the best mask we have
                mask[y0:y1, x0:x1] = 1
            stem = None
            x, y = clipping.findtext("center/x"), clipping.findtext("center/y")
            if clipping.findtext("center/hasNoCenter") == "false" and x and y:
                stem = (float(x), float(y))
            trusted = clipping.findtext("trusted/isTrusted") != "false"
            instances.append(Instance(
                source_label=source_label(clipping),
                bbox_xyxy=(int(x0), int(y0), int(x1), int(y1)),
                mask=mask.astype(bool),
                stem_xy=stem,
                ignore=len(points) < 3 or not trusted,
            ))
        crop, batch, _ = name.split("/")
        attributes = {"crop_field": crop.lower(), "batch": batch}
        if mm_per_px_estimate is not None:
            attributes["mm_per_px"] = mm_per_px_estimate
            attributes["mm_per_px_estimated"] = True
        yield ImageRecord(file_name=f"{name}.jpg", width=width, height=height,
                          group=batch_of(name), instances=instances, attributes=attributes)
