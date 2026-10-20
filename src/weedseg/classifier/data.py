"""Crop cache -> classifier samples (normalised image tensor, class index)."""

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

    With `augment` (training): random box jitter (`box_scale`, `box_shift`), flips, 90-degree
    rotations, colour jitter and, unless the background is kept, mask jitter
    (`mask_jitter_px`). Without: the plain crop, as at prediction.
    """

    def __init__(self, rows: list[dict[str, Any]], folder: Path, classes: list[str],
                 crops: dict[str, Any], augment: dict[str, Any] | None = None):
        index = {name: i for i, name in enumerate(classes)}
        unknown = sorted({row["category"] for row in rows} - set(index))
        if unknown:
            raise ValueError(f"Crops of classes the model doesn't have: {', '.join(unknown)}")
        self.rows, self.folder, self.crops, self.augment = rows, folder, crops, augment
        self.labels = [index[row["category"]] for row in rows]
        self.color_jitter = (v2.ColorJitter(**augment["color_jitter"])
                             if augment and augment.get("color_jitter") else None)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
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
        return to_input(crop, mask, self.crops, self.color_jitter), self.labels[i]


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


def jitter_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow (pixels > 0) or shrink (< 0) a mask, like the detector's rougher outlines."""
    if pixels == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(pixels) + 1,) * 2)
    operation = cv2.dilate if pixels > 0 else cv2.erode
    return operation(mask.astype(np.uint8), kernel) > 0


def sample_weights(labels: list[int], balance: str) -> list[float] | None:
    """Per-sample weights for a WeightedRandomSampler: `sqrt` and `inverse` draw classes in
    proportion to the square root of their size, or equally; `none` = None (plain shuffle)."""
    if balance not in BALANCE:
        raise ValueError(f"balance must be one of {BALANCE}, not {balance!r}")
    if balance == "none":
        return None
    counts = Counter(labels)
    power = 0.5 if balance == "sqrt" else 1.0
    return [counts[label] ** -power for label in labels]
