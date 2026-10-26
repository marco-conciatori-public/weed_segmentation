"""Two-stage prediction: plant detections (one-class Mask R-CNN) classified by species.

Each detection is cut out as the classifier was trained (weedseg.classifier.crops: same
margin, size and background treatment) and classified. The result keeps the detection's box
and mask and gets the species classes of the classifier, in the same COCO results format as
the species Mask R-CNN, so the same evaluation, decisions and error review apply:

- `score` and `plant_score` are the detector's score: how sure it is that there is a plant.
  The evaluation's score threshold and AP use it;
- `class_scores[c]` is the classifier's probability of c, given that it is a plant;
- `category_id` is the most probable class.

The two questions stay apart, so the decision rules can set a threshold for each (see
weedseg.decisions). Multiplying them would mix them, and the classifier's probabilities,
which label smoothing keeps under ~0.9, would then cap every score.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from weedseg.classifier.crops import FILL_RGB, cut_square, square_window
from weedseg.classifier.data import to_input
from weedseg.data.coco import decode_mask
from weedseg.maskrcnn.engine import CLASS_SCORE_MIN
from weedseg.maskrcnn.run import resolve_run_dir
from weedseg.paths import resolve


def detection_input(image: np.ndarray, detection: dict[str, Any],
                    crops: dict[str, Any]) -> torch.Tensor:
    """Classifier input for one detection on an RGB image."""
    window = square_window(detection["bbox"], crops["margin"])
    crop = cut_square(image, *window, crops["input_size"], FILL_RGB)
    mask = None
    if crops["background"] != "keep":
        full = decode_mask(detection, *image.shape[:2])
        if full is None:  # no mask: the box is the best we have
            x, y, w, h = detection["bbox"]
            full = np.zeros(image.shape[:2], bool)
            full[round(y):round(y + h), round(x):round(x + w)] = True
        mask = cut_square(full.astype(np.uint8), *window, crops["input_size"], nearest=True) > 0
    return to_input(crop, mask, crops)


def with_species(detection: dict[str, Any], probs: np.ndarray,
                 class_ids: list[int]) -> dict[str, Any]:
    """A plant detection relabelled with the classifier's probabilities (see module doc)."""
    plant = detection["score"]
    best = int(probs.argmax())
    return {**detection, "category_id": class_ids[best], "score": plant, "plant_score": plant,
            "class_scores": {str(class_ids[k]): round(float(probs[k]), 4)
                             for k in np.flatnonzero(probs >= CLASS_SCORE_MIN)}}


class DetectionCrops(Dataset):
    """Per image: the classifier inputs of its detections [n, 3, size, size]."""

    def __init__(self, images: list[dict[str, Any]], detections: dict[int, list[dict[str, Any]]],
                 image_dir: Path, crops: dict[str, Any]):
        self.images = [i for i in images if detections[i["id"]]]
        self.detections, self.image_dir, self.crops = detections, image_dir, crops

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        info = self.images[index]
        image = cv2.imread(str(self.image_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.image_dir / info["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return info["id"], torch.stack([detection_input(image, d, self.crops)
                                        for d in self.detections[info["id"]]])


@torch.no_grad()
def classify_detections(detections: list[dict[str, Any]], images: list[dict[str, Any]],
                        image_dir: Path, classifier: nn.Module, class_ids: list[int],
                        crops: dict[str, Any], device: torch.device, amp: bool,
                        batch_size: int = 128, min_plant_score: float = 0.0,
                        num_workers: int = 0) -> list[dict[str, Any]]:
    """Two-stage results: every detection with at least `min_plant_score`, classified."""
    by_image = defaultdict(list)
    for detection in detections:
        if detection["score"] >= min_plant_score:
            by_image[detection["image_id"]].append(detection)
    loader = DataLoader(DetectionCrops(images, by_image, image_dir, crops), batch_size=None,
                        num_workers=num_workers)
    classifier.eval()
    results = []
    for image_id, inputs in tqdm(loader, desc="classify", dynamic_ncols=True):
        probs = []
        for batch in inputs.split(batch_size):
            with torch.autocast(device.type, enabled=amp):
                logits = classifier(batch.to(device, non_blocking=True))
            probs.append(logits.float().softmax(1).cpu().numpy())
        results += [with_species(d, p, class_ids)
                    for d, p in zip(by_image[int(image_id)], np.concatenate(probs))]
    return results


def config_problems(plant: dict[str, Any], detector: dict[str, Any], classifier: dict[str, Any],
                    twostage: dict[str, Any]) -> list[str]:
    """Why the configs of the two-stage steps would work on different data (empty if they
    fit): the plant copy, the classifier and the prediction must use the same species data,
    and the detector the plant copy of it."""
    problems = []
    species = resolve(plant["source_dir"])
    for name, config in (("classifier", classifier), ("twostage", twostage)):
        if resolve(config["data"]["dir"]) != species:
            problems.append(f"data.dir of the {name} config ({config['data']['dir']}) is not "
                            f"the species data of plant_dataset.yaml ({plant['source_dir']})")
    plant_dir = resolve(detector["data"]["dir"])
    if plant_dir.parent != species.parent or not plant_dir.name.startswith("plant"):
        problems.append(f"data.dir of the detector config ({detector['data']['dir']}) is not "
                        f"a plant folder next to {plant['source_dir']}")
    return problems


def shared_run_dirs(classifiers: dict[str, dict[str, Any]]) -> list[str]:
    """Classifier configs ({path: config}) that would train the same run folder, e.g. a
    variant that kept its base config's run_name: its training would resume the other's
    finished run (and do nothing) instead of training anew. Empty if each has its own."""
    by_dir = defaultdict(list)
    for path, config in classifiers.items():
        by_dir[resolve_run_dir(config)].append(path)
    return [f"{' and '.join(paths)} train the same run, {run_dir}; give each its own run_name"
            for run_dir, paths in by_dir.items() if len(paths) > 1]
