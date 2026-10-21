"""COCO file -> torchvision detection samples (image tensor, target dict)."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from weedseg.data.coco import IGNORE_CATEGORY, decode_mask, drop_box_only
from weedseg.paths import resolve

# Training label of ignore regions. torchvision's samplers skip proposals and anchors labelled
# -1, so these regions count neither as plants nor as background (see maskrcnn.model).
IGNORE_LABEL = -1


def load_coco(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_masked_coco(path: Path) -> dict[str, Any]:
    """`load_coco` without the images labelled with boxes only, which Mask R-CNN can't learn
    from (weedseg.data.coco.drop_box_only), saying how many were left out. Training,
    evaluation and the error review all read their files this way, so they see the same
    images."""
    coco = load_coco(path)
    masked = drop_box_only(coco)
    left_out = len(coco["images"]) - len(masked["images"])
    if left_out:
        print(f"{path.name}: {left_out} of {len(coco['images'])} images left out "
              "(labelled with boxes only)")
    return masked


def model_categories(coco: dict[str, Any]) -> list[dict[str, Any]]:
    """The categories the model predicts, in label order (label = index + 1, 0 = background)."""
    return [c for c in coco["categories"] if c["name"] != IGNORE_CATEGORY]


def build_augmentation(config: dict[str, Any]) -> v2.Transform:
    """Training augmentation; boxes and masks follow the image. Top-down images have no
    preferred orientation, so both flips are valid."""
    transforms = []
    if config.get("horizontal_flip"):
        transforms.append(v2.RandomHorizontalFlip(config["horizontal_flip"]))
    if config.get("vertical_flip"):
        transforms.append(v2.RandomVerticalFlip(config["vertical_flip"]))
    if config.get("color_jitter"):
        transforms.append(v2.ColorJitter(**config["color_jitter"]))
    return v2.Compose(transforms)


class CocoInstanceDataset(Dataset):
    """Images of a COCO file with torchvision Mask R-CNN targets.

    Ignore regions (iscrowd, or the ignore category) get label IGNORE_LABEL. With
    `with_targets=False` only `image_id` is returned (for prediction).
    """

    def __init__(self, coco: dict[str, Any], augmentation: v2.Transform | None = None,
                 with_targets: bool = True, max_images: int | None = None, seed: int = 0):
        self.image_dir = resolve(coco["info"]["image_dir"])
        self.categories = model_categories(coco)
        self.labels = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
        self.images = list(coco["images"])
        if max_images is not None and max_images < len(self.images):
            self.images = random.Random(seed).sample(self.images, max_images)
        self.annotations = defaultdict(list)
        if with_targets:
            for annotation in coco["annotations"]:
                self.annotations[annotation["image_id"]].append(annotation)
        self.augmentation = augmentation
        self.with_targets = with_targets

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        info = self.images[index]
        path = self.image_dir / info["file_name"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = tv_tensors.Image(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                 .permute(2, 0, 1))
        if not self.with_targets:
            return v2.functional.to_dtype(image, torch.float32, scale=True), {"image_id": info["id"]}

        height, width = image.shape[-2:]
        boxes, labels, masks = [], [], []
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = annotation["bbox"]
            if w < 1 or h < 1:
                continue
            ignore = annotation.get("iscrowd") or annotation["category_id"] not in self.labels
            mask = decode_mask(annotation, height, width)
            if mask is None:  # no outline, e.g. an ignore region: its box
                mask = np.zeros((height, width), bool)
                mask[round(y):round(y + h), round(x):round(x + w)] = True
            boxes.append([x, y, x + w, y + h])
            labels.append(IGNORE_LABEL if ignore else self.labels[annotation["category_id"]])
            masks.append(mask)
        target = {
            "boxes": tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                              format="XYXY", canvas_size=(height, width)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(torch.from_numpy(np.stack(masks)) if masks
                                     else torch.zeros((0, height, width), dtype=torch.bool)),
            "image_id": info["id"],
        }
        if self.augmentation is not None:
            image, target = self.augmentation(image, target)
        target["masks"] = target["masks"].to(torch.uint8)
        return v2.functional.to_dtype(image, torch.float32, scale=True), target


def collate(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[list, list]:
    images, targets = zip(*batch)
    return list(images), list(targets)
