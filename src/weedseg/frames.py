"""Placing overlapping frames on each other from the images alone, for weedseg.overlap.

Imports neither torch nor the evaluation code: on Windows every worker process imports what its job
needs, and a worker takes about 0.3 GB of memory without torch, about 1.2 GB with it. See
docs/twostage.md (Overlapping frames).
"""

import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

REGISTERING = ("gaps", "side", "features", "ratio", "ransac_px")


def frame_order(name: str) -> list[Any]:
    """Sort key of a file name: numbers count as numbers, so frame 9 comes before frame 10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def frame_groups(images: list[dict[str, Any]], dataset: str) -> dict[str, list[dict[str, Any]]]:
    """A dataset's images per session, in the order they were taken."""
    groups = defaultdict(list)
    for info in images:
        if info["dataset"] == dataset:
            groups[info["group"]].append(info)
    return {group: sorted(items, key=lambda info: frame_order(info["file_name"]))
            for group, items in groups.items()}


def features(path: Path, settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None, float]:
    """SIFT points and descriptors of an image scaled so its longer side is at most `side`; the
    scale."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    scale = min(1.0, settings["side"] / max(image.shape))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points, described = cv2.SIFT_create(settings["features"]).detectAndCompute(small, None)
    return np.float32([point.pt for point in points]), described, scale


def register(a: tuple, b: tuple, settings: dict[str, Any]) -> tuple[np.ndarray, int] | None:
    """The similarity taking the full-size pixels of frame a to those of frame b, with its inliers;
    None if the two don't match."""
    if a[1] is None or b[1] is None or len(a[0]) < 8 or len(b[0]) < 8:
        return None
    good = [pair[0] for pair in cv2.BFMatcher().knnMatch(a[1], b[1], k=2)
            if len(pair) == 2 and pair[0].distance < settings["ratio"] * pair[1].distance]
    if len(good) < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        a[0][[match.queryIdx for match in good]], b[0][[match.trainIdx for match in good]],
        method=cv2.RANSAC, ransacReprojThreshold=settings["ransac_px"])
    if matrix is None:
        return None
    full = matrix.copy()
    full[:, :2] *= a[2] / b[2]
    full[:, 2] /= b[2]
    return full, int(inliers.sum())


def _register_group(job: tuple) -> list[dict[str, Any]]:
    """Register each frame of a session with the neighbours within `gaps` before it."""
    frames, settings = job
    kept, found = {}, {}
    gaps = settings["gaps"]
    for i, (image_id, path) in enumerate(frames):
        kept.pop(i - max(gaps) - 1, None)
        kept[i] = features(path, settings)
        for gap in gaps:
            if i - gap in kept:
                result = register(kept[i - gap], kept[i], settings)
                if result is not None:
                    found[frames[i - gap][0], image_id] = result
    return [{"a": a, "b": b, "matrix": matrix.tolist(), "inliers": inliers}
            for (a, b), (matrix, inliers) in found.items()]


def registrations(groups: dict[str, list[dict[str, Any]]], image_dir: Path,
                  settings: dict[str, Any], path: Path, num_workers: int) -> list[dict[str, Any]]:
    """The registered pairs of a dataset's sessions, kept at `path`: redone only if the frames or
    `REGISTERING` settings differ."""
    wanted = {"settings": {key: settings[key] for key in REGISTERING},
              "frames": {group: [info["file_name"] for info in items]
                         for group, items in groups.items()}}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {key: saved[key] for key in wanted} == wanted:
            return saved["pairs"]
    jobs = [([(info["id"], image_dir / info["file_name"]) for info in items], settings)
            for items in groups.values()]
    pairs = []
    progress = {"total": len(jobs), "desc": f"register {path.stem}", "dynamic_ncols": True}
    if num_workers > 0:
        with Pool(num_workers) as pool:
            for part in tqdm(pool.imap_unordered(_register_group, jobs), **progress):
                pairs += part
    else:
        for job in tqdm(jobs, **progress):
            pairs += _register_group(job)
    pairs.sort(key=lambda pair: (pair["a"], pair["b"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({**wanted, "pairs": pairs}), encoding="utf-8")
    partial.replace(path)
    return pairs


def frame_shift(matrix: list[list[float]], size: tuple[int, int]) -> float:
    """How far a frame's centre moved, as a share of its longer side."""
    width, height = size
    moved = np.asarray(matrix) @ np.array([width / 2, height / 2, 1.0])
    return float(np.hypot(moved[0] - width / 2, moved[1] - height / 2)) / max(size)


def usable(pair: dict[str, Any], size: tuple[int, int], settings: dict[str, Any]) -> bool:
    """A registered pair with enough inliers, about one scale, and a real move between the two."""
    scale = float(np.hypot(*np.asarray(pair["matrix"])[0, :2]))
    return (pair["inliers"] >= settings["min_inliers"]
            and abs(scale - 1) <= settings["max_scale_change"]
            and frame_shift(pair["matrix"], size) >= settings["min_shift"])
