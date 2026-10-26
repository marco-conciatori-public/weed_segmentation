"""Crop cache -> classifier samples (normalised image tensor, class index, allowed classes)."""

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from weedseg.classifier.crops import (
    FILL_RGB,
    apply_background,
    cache_size,
    cut_square,
    square_window,
)

# DINOv2 was trained with ImageNet normalisation.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BALANCE = ("none", "sqrt", "inverse")
# The coarsest detail the detail augmentation goes to, in px across the crop.
MIN_DETAIL_PX = 8


def to_input(crop: np.ndarray, mask: np.ndarray | None, crops: dict[str, Any],
             color_jitter: v2.Transform | None = None) -> torch.Tensor:
    """An RGB uint8 crop (input_size square) -> the classifier's input: colour jitter (if
    any), the background treatment of `crops`, ImageNet normalisation."""
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
    if color_jitter is not None:
        tensor = color_jitter(tensor)
    if crops["background"] != "keep":
        crop = apply_background(tensor.permute(1, 2, 0).numpy(), mask, crops["background"],
                                crops["dim_factor"])
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
    return (tensor.float() / 255 - MEAN) / STD


class CropDataset(Dataset):
    """Cached plant crops (see weedseg.classifier.crops) with their class index.

    A sample is (image, label, allowed): `label` is the class index, or -1 for a plant with
    a partial label (index column `allowed`); `allowed` marks the classes it may belong to
    (bool, one per class), its own class alone for a labelled plant.

    With `augment` (training): random box jitter (`box_scale`, `box_shift`), flips, 90-degree
    rotations, less detail (`detail_p`, `blur_p`, `jpeg_p`, see `lower_detail`), colour
    jitter and, unless the background is kept, mask jitter (`mask_jitter_px`). Without: the
    plain crop, as at prediction.
    """

    def __init__(self, rows: list[dict[str, Any]], folder: Path, classes: list[str],
                 crops: dict[str, Any], augment: dict[str, Any] | None = None):
        index = {name: i for i, name in enumerate(classes)}
        needed = {name for row in rows for name in row.get("allowed") or [row["category"]]}
        unknown = sorted(needed - set(index))
        if unknown:
            raise ValueError(f"Crops of classes the model doesn't have: {', '.join(unknown)}")
        self.rows, self.folder, self.crops, self.augment = rows, folder, crops, augment
        self.num_classes = len(classes)
        self.allowed = [[index[name] for name in row.get("allowed") or []] for row in rows]
        self.labels = [-1 if allowed else index[row["category"]]
                       for row, allowed in zip(rows, self.allowed)]
        self.color_jitter = (v2.ColorJitter(**augment["color_jitter"])
                             if augment and augment.get("color_jitter") else None)

    def __len__(self) -> int:
        return len(self.rows)

    def balance_keys(self) -> list[Any]:
        """What `sample_weights` balances: the class, or the set of allowed classes."""
        return [label if label >= 0 else tuple(allowed)
                for label, allowed in zip(self.labels, self.allowed)]

    def allowed_mask(self, i: int) -> torch.Tensor:
        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        mask[self.allowed[i] or [self.labels[i]]] = True
        return mask

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        stem = self.folder / str(self.rows[i]["annotation_id"])
        image = cv2.imread(f"{stem}.jpg", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"{stem}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        need_mask = self.crops["background"] != "keep"
        mask = cv2.imread(f"{stem}.png", cv2.IMREAD_GRAYSCALE) if need_mask else None

        # The cached crop is centred on the plant; the training crop is its central part
        # (1 / cache_context of the side), jittered when augmenting.
        size, augment = cache_size(self.crops), self.augment
        box_side = size / self.crops["cache_context"] / (1 + 2 * self.crops["margin"])
        box = [(size - box_side) / 2, (size - box_side) / 2, box_side, box_side]
        scale, shift = 1.0, (0.0, 0.0)
        if augment:
            scale = 1 + random.uniform(-augment["box_scale"], augment["box_scale"])
            shift = (random.uniform(-augment["box_shift"], augment["box_shift"]),
                     random.uniform(-augment["box_shift"], augment["box_shift"]))
        window = square_window(box, self.crops["margin"], scale, shift)
        crop = cut_square(image, *window, self.crops["input_size"], FILL_RGB)
        if mask is not None:
            mask = cut_square(mask, *window, self.crops["input_size"], nearest=True) > 127
            if augment and augment.get("mask_jitter_px"):
                mask = jitter_mask(mask, random.randint(-augment["mask_jitter_px"],
                                                        augment["mask_jitter_px"]))
        if augment:
            crop, mask = flip_rotate(crop, mask, augment)
            crop = lower_detail(crop, augment, *self.detail(i, window[2], scale))
        return (to_input(crop, mask, self.crops, self.color_jitter), self.labels[i],
                self.allowed_mask(i))

    def detail(self, i: int, window_px: float, scale: float) -> tuple[float | None, float | None]:
        """The ground the training crop covers (mm) and the mm/px of the detail it holds: the
        source image's, or coarser when the cache or the resizing to input_size kept fewer
        pixels than the source had. (None, None) without a known size or resolution."""
        row = self.rows[i]
        if not row.get("size_mm") or not row.get("mm_per_px"):
            return None, None
        window_mm = row["size_mm"] * (1 + 2 * self.crops["margin"]) * scale
        pixels = min(window_mm / row["mm_per_px"], window_px, self.crops["input_size"])
        return window_mm, window_mm / pixels


def flip_rotate(crop: np.ndarray, mask: np.ndarray | None,
                augment: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """Random flips and 90-degree rotations: top-down images have no preferred orientation."""
    def both(function):
        return function(crop), (function(mask) if mask is not None else None)

    if augment.get("flips") and random.random() < 0.5:
        crop, mask = both(np.fliplr)
    if augment.get("rotate90"):
        k = random.randrange(4)
        crop, mask = both(lambda a: np.rot90(a, k))
    return crop, mask


def lower_detail(crop: np.ndarray, augment: dict[str, Any], window_mm: float | None,
                 own_mm_per_px: float | None) -> np.ndarray:
    """Random loss of detail, as from a coarser or worse camera, so the classifier can't tell
    the source dataset by its sharpness (OPPD's 0.15 mm/px against PhenoBench's 1 mm/px):

    - with probability `detail_p`, the crop (covering `window_mm` of ground, with detail of
      `own_mm_per_px`) is shrunk to the detail of a random mm/px between its own and
      `max_mm_per_px` (log-uniform) and enlarged back; skipped without a known resolution;
    - with probability `blur_p`, a Gaussian blur of sigma drawn from `blur_sigma` (px);
    - with probability `jpeg_p`, JPEG compression at a quality drawn from `jpeg_quality`.
    """
    size = crop.shape[0]
    crop = np.ascontiguousarray(crop)
    if window_mm and own_mm_per_px:  # the crop holds no more detail than its pixels
        own_mm_per_px = max(own_mm_per_px, window_mm / size)
    if (window_mm and own_mm_per_px and own_mm_per_px < augment.get("max_mm_per_px", 0)
            and random.random() < augment.get("detail_p", 0)):
        target = math.exp(random.uniform(math.log(own_mm_per_px),
                                         math.log(augment["max_mm_per_px"])))
        pixels = max(MIN_DETAIL_PX, round(window_mm / target))
        if pixels < size:
            small = cv2.resize(crop, (pixels, pixels), interpolation=cv2.INTER_AREA)
            crop = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
    if random.random() < augment.get("blur_p", 0):
        crop = cv2.GaussianBlur(crop, (0, 0), random.uniform(*augment["blur_sigma"]))
    if random.random() < augment.get("jpeg_p", 0):
        quality = random.randint(*augment["jpeg_quality"])
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, quality])
        crop = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return crop


def jitter_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow (pixels > 0) or shrink (< 0) a mask, like the detector's rougher outlines."""
    if pixels == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    operation = cv2.dilate if pixels > 0 else cv2.erode
    return operation(mask.astype(np.uint8), kernel) > 0


def sample_weights(labels: list[Any], balance: str) -> list[float] | None:
    """Per-sample weights for a WeightedRandomSampler: `sqrt` and `inverse` draw classes in
    proportion to the square root of their size, or equally; `none` = None (plain shuffle).
    `labels` may be any hashable keys (CropDataset.balance_keys)."""
    if balance not in BALANCE:
        raise ValueError(f"balance must be one of {BALANCE}, not {balance!r}")
    if balance == "none":
        return None
    counts = Counter(labels)
    power = 0.5 if balance == "sqrt" else 1.0
    return [counts[label] ** -power for label in labels]
