"""Training and prediction loops for the Mask R-CNN baseline."""

import math
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from tqdm import tqdm


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_atomic(data: dict[str, Any], path: Path) -> None:
    """torch.save via a temporary file, so an interruption never leaves a broken checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR factor: linear warm-up from 0.001 to 1, then cosine decay to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return 0.001 + (1 - 0.001) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, scaler, device: torch.device,
                    amp: bool, grad_clip: float | None, epoch: int,
                    on_step: Callable[[dict[str, float]], None] | None = None) -> dict[str, float]:
    """One pass over `loader`; returns the mean of each loss. `on_step` gets each step's
    losses and learning rate (for logging)."""
    model.train()
    sums, steps = defaultdict(float), 0
    progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
    for images, targets in progress:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in target.items()} for target in targets]
        with torch.autocast(device.type, enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss is {loss.item()} at epoch {epoch}: "
                                     f"{ {k: v.item() for k, v in losses.items()} }")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        values = {k: v.item() for k, v in losses.items()} | {"loss": loss.item()}
        for k, v in values.items():
            sums[k] += v
        steps += 1
        progress.set_postfix(loss=f"{sums['loss'] / steps:.3f}")
        if on_step is not None:
            on_step(values | {"lr": optimizer.param_groups[0]["lr"]})
    return {k: v / max(1, steps) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader: DataLoader, categories: list[dict[str, Any]], device: torch.device,
            amp: bool, mask_threshold: float = 0.5) -> list[dict[str, Any]]:
    """COCO results (bbox x, y, w, h, score, RLE segmentation) for every image in `loader`."""
    model.eval()
    category_ids = [c["id"] for c in categories]  # label k -> category_ids[k - 1]
    results = []
    for images, targets in tqdm(loader, desc="predict", dynamic_ncols=True):
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(device.type, enabled=amp):
            outputs = model(images)
        for output, target in zip(outputs, targets):
            if len(output["boxes"]) == 0:
                continue
            boxes = output["boxes"].float().cpu().numpy()
            scores = output["scores"].float().cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = (output["masks"][:, 0] > mask_threshold).cpu().numpy()
            rles = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
            for box, score, label, rle in zip(boxes, scores, labels, rles):
                rle["counts"] = rle["counts"].decode("ascii")
                x0, y0, x1, y1 = box.tolist()
                results.append({"image_id": target["image_id"],
                                "category_id": category_ids[label - 1],
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "score": float(score), "segmentation": rle})
    return results
