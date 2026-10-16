"""Reader for the CropAndWeed dataset (Steininger et al., WACV 2023).

Raw layout after download (see docs/datasets/cropandweed.md):
    images/<name>.jpg                  1920x1088 RGB
    bboxes/<name>.csv                  no header: left, top, right, bottom, label_id, stem_x, stem_y
    labelIds/CropAndWeed/<name>.png    uint8 semantic mask with the same label ids (0 = soil)
    params/<name>.csv                  header + one row: moisture, soil, lighting, separability

<name> is "<subset>-<session>-<image>", e.g. "vwg-0328-0004"; subset "ave" is the
Application Set (commercial fields), "vwg" the Experimental Set (cultivated plots).
"""

import csv
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from weedseg.data.coco import ImageRecord, Instance

VEGETATION_LABEL = 255  # the dataset's fallback for tiny or unidentifiable plants
SUBSETS = {"ave": "application", "vwg": "experimental"}
GROWTH_STAGES = {  # label id -> growth stage, for the crops annotated by stage
    2: "two-leaf", 3: "four-leaf", 4: "six-leaf", 5: "eight-leaf", 6: "max",
    8: "two-leaf", 9: "four-leaf", 10: "six-leaf", 11: "eight-leaf", 12: "max",
}
PARAM_LEVELS = {
    "moisture": ["dry", "medium", "wet"],
    "soil": ["fine", "medium", "coarse"],
    "lighting": ["sunny", "diffuse"],
    "separability": ["easy", "medium", "hard"],
}


def read_boxes(path: Path) -> list[tuple[int, ...]]:
    """Rows of (left, top, right, bottom, label_id, stem_x, stem_y); coordinates inclusive."""
    with open(path, newline="", encoding="utf-8") as f:
        return [tuple(int(v) for v in row) for row in csv.reader(f) if row]


def read_params(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    return {key: PARAM_LEVELS[key][int(value)] for key, value in row.items()}


def instance_masks(semantic: np.ndarray, boxes: list[tuple[int, ...]]) -> list[np.ndarray]:
    """Split the semantic mask into one mask per box.

    The dataset has semantic masks only. A box's instance is the pixels inside it that carry
    its label. Where boxes of the same label overlap, each shared pixel goes to the box with
    the nearest stem point (ties go to the earlier box). Returns full-size bool masks,
    possibly empty.
    """
    height, width = semantic.shape
    masks = []
    for i, (left, top, right, bottom, label, stem_x, stem_y) in enumerate(boxes):
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(right + 1, width), min(bottom + 1, height)
        crop = semantic[y0:y1, x0:x1] == label
        ys, xs = np.mgrid[y0:y1, x0:x1]
        own_distance = (xs - stem_x) ** 2 + (ys - stem_y) ** 2
        for j, (l2, t2, r2, b2, label2, sx2, sy2) in enumerate(boxes):
            if j == i or label2 != label or l2 > right or r2 < left or t2 > bottom or b2 < top:
                continue
            inside = (xs >= l2) & (xs <= r2) & (ys >= t2) & (ys <= b2)
            other_distance = (xs - sx2) ** 2 + (ys - sy2) ** 2
            closer = other_distance < own_distance if j > i else other_distance <= own_distance
            crop &= ~(inside & closer)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = crop
        masks.append(mask)
    return masks


def unboxed_regions(semantic: np.ndarray, masks: list[np.ndarray],
                    min_area: int) -> list[Instance]:
    """Ignore regions for labelled pixels that belong to no box.

    Some semantic masks contain plants without a box (e.g. dense patches). Left as they are,
    a model would learn them as background. Each connected region of at least `min_area`
    pixels becomes an ignore instance labelled with its most frequent label.
    """
    covered = np.zeros(semantic.shape, dtype=bool)
    for mask in masks:
        covered |= mask
    leftover = ((semantic > 0) & ~covered).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(leftover, connectivity=8)
    regions = []
    for k in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[k])
        if area < min_area:
            continue
        mask = components == k
        labels, counts = np.unique(semantic[mask], return_counts=True)
        regions.append(Instance(source_label=str(labels[counts.argmax()]),
                                bbox_xyxy=(x, y, x + w, y + h), mask=mask, ignore=True,
                                attributes={"unboxed": True}))
    return regions


def session_of(name: str) -> str:
    """Recording session of an image name, e.g. "vwg-0328" for "vwg-0328-0004"."""
    return name.rsplit("-", 1)[0]


def image_names(raw_dir: Path, keep_negative_images: bool = True,
                require_image_file: bool = True) -> list[str]:
    """Names of the annotated images; negatives are images with an empty box file."""
    names = []
    for box_file in sorted((raw_dir / "bboxes").glob("*.csv")):
        if require_image_file and not (raw_dir / "images" / f"{box_file.stem}.jpg").exists():
            continue
        if not keep_negative_images and box_file.stat().st_size == 0:
            continue
        names.append(box_file.stem)
    return names


def image_labels(raw_dir: Path, names: list[str]) -> list[set[str]]:
    """Source labels of the boxes of each image (reads only the box files, so it is fast)."""
    return [{str(box[4]) for box in read_boxes(raw_dir / "bboxes" / f"{name}.csv")}
            for name in names]


def iter_records(raw_dir: Path, names: list[str], with_masks: bool = True,
                 unboxed_min_area: int | None = 256,
                 mm_per_px_estimate: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per image name, labels as the dataset's numeric ids.

    With masks, labelled regions without a box are added as ignore instances unless
    `unboxed_min_area` is None.
    """
    for name in names:
        boxes = read_boxes(raw_dir / "bboxes" / f"{name}.csv")
        semantic = cv2.imread(str(raw_dir / "labelIds" / "CropAndWeed" / f"{name}.png"),
                              cv2.IMREAD_UNCHANGED)
        height, width = semantic.shape
        masks = instance_masks(semantic, boxes) if with_masks else [None] * len(boxes)

        instances = []
        for (left, top, right, bottom, label, stem_x, stem_y), mask in zip(boxes, masks):
            instance = Instance(
                source_label=str(label),
                bbox_xyxy=(left, top, right + 1, bottom + 1),
                mask=mask,
                stem_xy=(stem_x, stem_y),
            )
            if label in GROWTH_STAGES:
                instance.attributes["growth_stage"] = GROWTH_STAGES[label]
            if mask is not None and not mask.any():
                # A few boxes have no pixel of their label; keep them as ignore regions.
                instance.mask[top:bottom + 1, left:right + 1] = True
                instance.ignore = True
                instance.attributes["mask_from_box"] = True
            instances.append(instance)
        if with_masks and unboxed_min_area is not None:
            instances += unboxed_regions(semantic, masks, unboxed_min_area)

        subset = name.split("-", 1)[0]
        attributes = {"subset": SUBSETS[subset], "session": session_of(name),
                      **read_params(raw_dir / "params" / f"{name}.csv")}
        if mm_per_px_estimate is not None:
            attributes["mm_per_px"] = mm_per_px_estimate
            attributes["mm_per_px_estimated"] = True
        yield ImageRecord(file_name=f"{name}.jpg", width=width, height=height,
                          group=attributes["session"], instances=instances,
                          attributes=attributes)
