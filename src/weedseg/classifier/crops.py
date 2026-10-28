"""Square crops around plants, for the species classifier, and the crop cache.

A plant's crop is a square centred on its box, with `margin` times the box's longer side of
context on each side. The same geometry serves training (cut from the cache) and the
two-stage prediction (cut from the full image around each detection).

The cache holds every labelled plant of a species split, and every hidden plant with a
partial label (`allowed_classes`, see weedseg.data.combine), cut once from the full images
with `cache_context` times the crop's side (room for box jitter in training) and resized to
`input_size * cache_context` px: <cache dir>/<split>/<annotation id>.jpg, the plant's mask
as <annotation id>.png, and index_<split>.csv. A partly labelled plant has the category
`ignore` there and its classes in the `allowed` column (separated by spaces).
"""

import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from weedseg.data.coco import IGNORE_CATEGORY, decode_mask

# Fill for pixels outside the image (and blanked background): the ImageNet mean, which
# the normalisation of DINOv2 and DINOv3 maps to 0.
FILL_RGB = (124, 116, 104)
BACKGROUNDS = ("keep", "dim", "blank")
INDEX_COLUMNS = ["annotation_id", "image_id", "category", "eppo", "size_mm", "mm_per_px",
                 "dataset", "allowed"]


def square_window(box: list[float], margin: float, scale: float = 1.0,
                  shift: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """Centre x, centre y and side of the crop around a box (x y w h): the box's longer side
    plus `margin` times it on each side, times `scale`; `shift` moves the centre by that
    fraction of the longer side."""
    x, y, w, h = box
    longer = max(w, h)
    return (x + w / 2 + shift[0] * longer, y + h / 2 + shift[1] * longer,
            longer * (1 + 2 * margin) * scale)


def cut_square(image: np.ndarray, cx: float, cy: float, side: float, size: int,
               fill: int | tuple[int, ...] = 0, nearest: bool = False) -> np.ndarray:
    """The square window (centre, side in px) of `image`, resized to size x size; parts
    outside the image get `fill`. `nearest` for masks."""
    side_px = max(1, round(side))
    x0, y0 = round(cx - side_px / 2), round(cy - side_px / 2)
    x1, y1 = x0 + side_px, y0 + side_px
    height, width = image.shape[:2]
    patch = image[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    pad = (max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width))
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_CONSTANT, value=fill)
    if patch.shape[:2] != (size, size):
        if nearest:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_AREA if side_px > size else cv2.INTER_LINEAR
        patch = cv2.resize(patch, (size, size), interpolation=interpolation)
    return patch


def apply_background(crop: np.ndarray, mask: np.ndarray | None, background: str,
                     dim_factor: float) -> np.ndarray:
    """`keep` the crop as it is, `dim` the pixels outside the plant's mask to `dim_factor` of
    their brightness, or `blank` them to FILL_RGB."""
    if background == "keep":
        return crop
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, not {background!r}")
    outside = ~mask.astype(bool)
    crop = crop.copy()
    if background == "dim":
        crop[outside] = (crop[outside] * dim_factor).astype(crop.dtype)
    else:
        crop[outside] = FILL_RGB
    return crop


def cache_dir(data_dir: Path, settings: dict[str, Any]) -> Path:
    """Cache folder of a species data folder for the crop settings (`margin`, `input_size`,
    `cache_context`): crops made with other settings live next to it."""
    return (data_dir / f"crops_margin{settings['margin']:g}_size{settings['input_size']}"
                       f"_context{settings['cache_context']:g}")


def cache_size(settings: dict[str, Any]) -> int:
    return round(settings["input_size"] * settings["cache_context"])


def plants_to_cache(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Labelled plants (not ignore regions) and plants with a partial label, per image id."""
    ignore_ids = {c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY}
    plants = defaultdict(list)
    for annotation in coco["annotations"]:
        labelled = not annotation.get("iscrowd") and annotation["category_id"] not in ignore_ids
        if labelled or annotation.get("allowed_classes"):
            plants[annotation["image_id"]].append(annotation)
    return plants


def index_row(annotation: dict[str, Any], info: dict[str, Any],
              names: dict[int, str]) -> dict[str, Any]:
    """The index row of a cached plant; `info` is its image."""
    mm_per_px = info.get("mm_per_px")
    return {
        "annotation_id": annotation["id"], "image_id": info["id"],
        "category": names[annotation["category_id"]], "eppo": annotation.get("eppo", ""),
        "size_mm": round(max(annotation["bbox"][2:]) * mm_per_px, 1) if mm_per_px else "",
        "mm_per_px": mm_per_px or "",
        "dataset": info.get("dataset", ""),
        "allowed": " ".join(annotation.get("allowed_classes", [])),
    }


def index_is_current(path: Path, coco: dict[str, Any]) -> bool:
    """Whether the index at `path` lists exactly the plants `coco` has to cache, with the
    same class, code, size, resolution and dataset. An index made from another version of
    the data (e.g. before a dataset was added or re-converted) is not."""
    images = {info["id"]: info for info in coco["images"]}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    expected = [{key: str(value) for key, value in index_row(a, images[image_id], names).items()}
                for image_id, annotations in plants_to_cache(coco).items() for a in annotations]
    expected.sort(key=lambda row: int(row["annotation_id"]))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f)) == expected


def _cache_image(job: tuple) -> list[dict[str, Any]]:
    """Write the crops of one image; returns their index rows."""
    info, annotations, image_dir, out_dir, settings, names = job
    image = cv2.imread(str(image_dir / info["file_name"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_dir / info["file_name"])
    height, width = image.shape[:2]
    size = cache_size(settings)
    fill_bgr = FILL_RGB[::-1]
    rows = []
    for annotation in annotations:
        cx, cy, side = square_window(annotation["bbox"], settings["margin"],
                                     settings["cache_context"])
        mask = decode_mask(annotation, height, width)
        if mask is None:  # box-only plant: the box is the best mask we have
            x, y, w, h = annotation["bbox"]
            mask = np.zeros((height, width), bool)
            mask[round(y):round(y + h), round(x):round(x + w)] = True
        crop = cut_square(image, cx, cy, side, size, fill_bgr)
        crop_mask = cut_square(mask.astype(np.uint8) * 255, cx, cy, side, size, nearest=True)
        stem = out_dir / str(annotation["id"])
        cv2.imwrite(f"{stem}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, settings["cache_quality"]])
        cv2.imwrite(f"{stem}.png", crop_mask)
        rows.append(index_row(annotation, info, names))
    return rows


def write_cache(coco: dict[str, Any], image_dir: Path, out_dir: Path, settings: dict[str, Any],
                num_workers: int = 0) -> Path:
    """Cut every plant of `coco` to cache (`plants_to_cache`) into `out_dir`; returns the
    index CSV path (`out_dir`'s parent / index_<out_dir name>.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    plants = plants_to_cache(coco)
    jobs = [(info, plants[info["id"]], image_dir, out_dir, settings, names)
            for info in coco["images"] if plants[info["id"]]]
    rows = []
    progress = {"total": len(jobs), "desc": f"crops {out_dir.name}", "dynamic_ncols": True}
    if num_workers > 0:
        with Pool(num_workers) as pool:
            for image_rows in tqdm(pool.imap_unordered(_cache_image, jobs, chunksize=4),
                                   **progress):
                rows += image_rows
    else:
        for job in tqdm(jobs, **progress):
            rows += _cache_image(job)
    rows.sort(key=lambda row: row["annotation_id"])
    index = out_dir.parent / f"index_{out_dir.name}.csv"
    with open(index, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return index


def read_index(path: Path) -> list[dict[str, Any]]:
    """Index rows, `allowed` as a list (empty for a plant labelled with its class); sizes
    and resolutions as floats or None."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["annotation_id"], row["image_id"] = int(row["annotation_id"]), int(row["image_id"])
        for key in ("size_mm", "mm_per_px"):
            row[key] = float(row[key]) if row.get(key) else None
        row["allowed"] = (row.get("allowed") or "").split()
    return rows
