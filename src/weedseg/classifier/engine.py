"""Training and prediction loops for the species classifier."""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler, scaler,
                    criterion: nn.Module, device: torch.device, amp: bool,
                    grad_clip: float | None, epoch: int,
                    on_step: Callable[[dict[str, float]], None] | None = None) -> dict[str, float]:
    """One pass over `loader`; returns the mean loss and the training accuracy."""
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss is {loss.item()} at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
        progress.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{correct / seen:.3f}")
        if on_step is not None:
            on_step({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
    return {"loss": loss_sum / max(1, seen), "accuracy": correct / max(1, seen)}


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                  amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities [samples, classes] and the labels of every sample in `loader`."""
    model.eval()
    probs, labels = [], []
    for images, batch_labels in tqdm(loader, desc="predict", dynamic_ncols=True):
        with torch.autocast(device.type, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu().numpy())
        labels.append(np.asarray(batch_labels))
    return np.concatenate(probs), np.concatenate(labels)


def accuracy_metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    """Top-1 accuracy, and balanced accuracy (mean of the per-class recalls, over the classes
    present)."""
    predicted = probs.argmax(1)
    recalls = [float((predicted[labels == k] == k).mean())
               for k in range(num_classes) if (labels == k).any()]
    return {"accuracy": float((predicted == labels).mean()) if len(labels) else 0.0,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0}
