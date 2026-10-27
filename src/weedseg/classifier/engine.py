"""Training and prediction loops for the species classifier."""

from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class PartialLabelLoss(nn.Module):
    """Cross-entropy for samples that may belong to any of a set of classes: minus the log of
    the probability the model gives to the whole set (`allowed`, bool [samples, classes]).
    For a set of one class it is the usual cross-entropy, so labelled and partly labelled
    plants share one loss. A partly labelled plant thus learns only that it isn't any of
    the other classes, e.g. a PhenoBench weed that it isn't sugar beet.

    Label smoothing as in nn.CrossEntropyLoss: (1 - label_smoothing) times this loss, plus
    label_smoothing times the mean over all classes of -log p.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        log_probs = logits.float().log_softmax(1)
        set_loss = -torch.logsumexp(log_probs.masked_fill(~allowed, float("-inf")), 1)
        smoothing = -log_probs.mean(1)
        return ((1 - self.label_smoothing) * set_loss + self.label_smoothing * smoothing).mean()


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler, scaler,
                    criterion: nn.Module, device: torch.device, amp: bool,
                    grad_clip: float | None, epoch: int,
                    on_step: Callable[[dict[str, float]], None] | None = None,
                    partial: dict[str, Any] | None = None, save_every: int | None = None,
                    save: Callable[[dict[str, Any]], None] | None = None) -> dict[str, float]:
    """One pass over `loader`; returns the mean loss and the training accuracy.

    Batches are (images, labels) or, from a CropDataset, (images, labels, allowed); the
    criterion then gets `allowed` (PartialLabelLoss). The accuracy leaves out samples with a
    partial label (label -1).

    Mid-epoch checkpoints as in weedseg.maskrcnn.engine.train_one_epoch: every `save_every`
    batches, `save` gets the state of the epoch so far, which `partial` takes back."""
    model.train()
    partial = partial or {"batches": 0, "loss_sum": 0.0, "correct": 0, "seen": 0,
                          "labelled": 0}
    loss_sum, correct, seen = partial["loss_sum"], partial["correct"], partial["seen"]
    labelled = partial.get("labelled", seen)
    batches = partial["batches"]
    progress = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True, initial=batches,
                    total=batches + len(loader))
    for images, labels, *allowed in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target = allowed[0].to(device, non_blocking=True) if allowed else labels
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, target)
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
        correct += (logits.argmax(1) == labels).sum().item()  # never true for label -1
        seen += len(labels)
        labelled += (labels >= 0).sum().item()
        progress.set_postfix(loss=f"{loss_sum / seen:.3f}",
                             acc=f"{correct / max(1, labelled):.3f}")
        if on_step is not None:
            on_step({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
        batches += 1
        if save_every and save is not None and batches % save_every == 0:
            save({"batches": batches, "loss_sum": loss_sum, "correct": correct, "seen": seen,
                  "labelled": labelled})
    return {"loss": loss_sum / max(1, seen), "accuracy": correct / max(1, labelled)}


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device,
                  amp: bool) -> tuple[np.ndarray, np.ndarray]:
    """Softmax probabilities [samples, classes] and the labels of every sample in `loader`
    (-1 for a partial label)."""
    model.eval()
    probs, labels = [], []
    for images, batch_labels, *_ in tqdm(loader, desc="predict", dynamic_ncols=True):
        with torch.autocast(device.type, enabled=amp):
            logits = model(images.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu().numpy())
        labels.append(np.asarray(batch_labels))
    return np.concatenate(probs), np.concatenate(labels)


def accuracy_metrics(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    """Top-1 accuracy, and balanced accuracy (mean of the per-class recalls, over the classes
    present). Samples with a partial label (-1) are left out."""
    probs, labels = probs[labels >= 0], labels[labels >= 0]
    predicted = probs.argmax(1)
    recalls = [float((predicted[labels == k] == k).mean())
               for k in range(num_classes) if (labels == k).any()]
    return {"accuracy": float((predicted == labels).mean()) if len(labels) else 0.0,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0}


def class_by_dataset(true: list[str], predicted: list[str], datasets: list[str],
                     train_counts: Counter, other: str, cross_share: float
                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Accuracy per class and dataset, for the classes found in more than one dataset (in
    training or in these crops), and the accuracy on cross-dataset crops.

    `train_counts` counts the training crops per (class, dataset). Each row: the class, the
    dataset, its crops, their accuracy, the class's training crops from that dataset and
    their share of all its training crops, and its most frequent wrong answer. A crop is
    cross-dataset when under `cross_share` of its class's training crops come from its own
    dataset: its species was learnt mostly from other datasets. `other` (other_plant) is
    left out of that, since it holds different species in each dataset.
    """
    cells = defaultdict(list)  # (class, dataset) -> predictions
    for label, guess, dataset in zip(true, predicted, datasets):
        cells[label, dataset].append(guess)
    class_total = Counter()
    sources = defaultdict(set)
    for (label, dataset), count in train_counts.items():
        class_total[label] += count
        sources[label].add(dataset)
    for label, dataset in cells:
        sources[label].add(dataset)

    def share(label: str, dataset: str) -> float:
        return train_counts[label, dataset] / class_total[label] if class_total[label] else 0.0

    rows = []
    for (label, dataset), guesses in sorted(cells.items()):
        if len(sources[label]) < 2:
            continue
        wrong = Counter(g for g in guesses if g != label).most_common(1)
        rows.append({"category": label, "dataset": dataset, "crops": len(guesses),
                     "accuracy": sum(g == label for g in guesses) / len(guesses),
                     "train_crops": train_counts[label, dataset],
                     "train_share": round(share(label, dataset), 4),
                     "most_wrong": (f"{wrong[0][0]} {wrong[0][1] / len(guesses):.0%}"
                                    if wrong else "")})
    cross = [label == guess for label, guess, dataset in zip(true, predicted, datasets)
             if label != other and share(label, dataset) < cross_share]
    summary = {"cross_dataset_crops": len(cross),
               "cross_dataset_accuracy": sum(cross) / len(cross) if cross else None}
    return rows, summary


def last_below_best(records: list[dict[str, Any]], tolerance: float) -> str | None:
    """A warning when the last epoch's val accuracy (metrics.jsonl records) is more than
    `tolerance` below the best epoch's, e.g. from overfitting; None otherwise."""
    if not records:
        return None
    best, last = max(records, key=lambda r: r["val_accuracy"]), records[-1]
    gap = best["val_accuracy"] - last["val_accuracy"]
    if gap <= tolerance:
        return None
    return (f"Warning: the last epoch ({last['epoch']}) has val accuracy "
            f"{last['val_accuracy']:.3f}, {gap:.1%} below the best (epoch {best['epoch']}, "
            f"{best['val_accuracy']:.3f}). Evaluation and prediction use last.pt by default; "
            "evaluate best.pt too (evaluate.checkpoint: best) before relying on it.")


def label_fit(predicted: np.ndarray, allowed: list[list[int]]) -> float | None:
    """Share of partly labelled samples whose predicted class is one of their allowed classes
    (e.g. of PhenoBench's weeds, those not called sugar beet); None without any."""
    if not len(allowed):
        return None
    return float(np.mean([guess in classes for guess, classes in zip(predicted, allowed)]))
