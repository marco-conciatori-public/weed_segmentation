"""Training and prediction loops for the Mask R-CNN baseline."""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

CLASS_SCORE_MIN = 0.001  # class probabilities kept per detection in predictions


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


class EpochSampler(Sampler[int]):
    """Random order of a dataset, fixed by (seed, epoch), that can start part-way through an
    epoch: a run resumed from a checkpoint saved mid-epoch gets the rest of that epoch's
    samples. With `weights`, samples are drawn with replacement in proportion to them (like
    WeightedRandomSampler). Call set_epoch before each epoch."""

    def __init__(self, size: int, seed: int, weights: Any = None):
        self.size, self.seed = size, seed
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.double)
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        """Order of `epoch`, without its first `skip` samples."""
        self.epoch, self.skip = epoch, skip

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed * 100_000 + self.epoch)
        if self.weights is None:
            order = torch.randperm(self.size, generator=generator)
        else:
            order = torch.multinomial(self.weights, self.size, replacement=True,
                                      generator=generator)
        return iter(order[self.skip:].tolist())

    def __len__(self) -> int:
        return self.size - self.skip


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, scaler, device: torch.device,
                    amp: bool, grad_clip: float | None, epoch: int,
                    on_step: Callable[[dict[str, float]], None] | None = None,
                    partial: dict[str, Any] | None = None, save_every: int | None = None,
                    save: Callable[[dict[str, Any]], None] | None = None) -> dict[str, float]:
    """One pass over `loader`; returns the mean of each loss. `on_step` gets each step's
    losses and learning rate (for logging).

    Mid-epoch checkpoints: every `save_every` batches, `save` gets the state of the epoch so
    far ({"batches", "sums", "steps"}); passed back as `partial` (with a loader that skips
    those batches), the epoch continues where it was saved."""
    model.train()
    partial = partial or {"batches": 0, "sums": {}, "steps": 0}
    sums, steps = defaultdict(float, partial["sums"]), partial["steps"]
    batches = partial["batches"]
    progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True, initial=batches,
                    total=batches + len(loader))
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
        batches += 1
        if save_every and save is not None and batches % save_every == 0:
            save({"batches": batches, "sums": dict(sums), "steps": steps})
    return {k: v / max(1, steps) for k, v in sums.items()}


@torch.no_grad()
def predict(model, loader: DataLoader, categories: list[dict[str, Any]], device: torch.device,
            amp: bool, mask_threshold: float = 0.5) -> list[dict[str, Any]]:
    """COCO results (bbox x, y, w, h, score, RLE segmentation) for every image in `loader`.

    Each result also has `class_scores`, {category id (str): probability} of the classes the
    model gave at least CLASS_SCORE_MIN (the rest, background included, is left out), for
    decisions on the role rather than the species (see weedseg.decisions).
    """
    model.eval()
    category_ids = [c["id"] for c in categories]  # label k -> category_ids[k - 1]
    results = []
    for images, targets in tqdm(loader, desc="predict", dynamic_ncols=True):
        images = [image.to(device, non_blocking=True) for image in images]
        with torch.autocast(device.type, enabled=amp):
            outputs = model(images)
        for output, target, probs in zip(outputs, targets, model.roi_heads.class_scores):
            if len(output["boxes"]) == 0:
                continue
            boxes = output["boxes"].float().cpu().numpy()
            scores = output["scores"].float().cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = (output["masks"][:, 0] > mask_threshold).cpu().numpy()
            probs = probs[:, 1:].float().cpu().numpy()  # without background
            rles = mask_utils.encode(np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8)))
            for box, score, label, rle, p in zip(boxes, scores, labels, rles, probs):
                rle["counts"] = rle["counts"].decode("ascii")
                x0, y0, x1, y1 = box.tolist()
                results.append({"image_id": target["image_id"],
                                "category_id": category_ids[label - 1],
                                "bbox": [x0, y0, x1 - x0, y1 - y0],
                                "score": float(score), "segmentation": rle,
                                "class_scores": {str(category_ids[k]): round(float(p[k]), 4)
                                                 for k in np.flatnonzero(p >= CLASS_SCORE_MIN)}})
    return results
