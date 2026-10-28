"""Weed species recognition in top-down field images: a two-stage model.

1. A Mask R-CNN (torchvision, ResNet-50 FPN v2) finds and outlines every plant.
2. A DINOv2 ViT-B/14 classifier (timm) names the species of each plant, from a square crop
   around it.

    from weed_model import WeedModel

    model = WeedModel()                       # this folder; the GPU when there is one
    plants = model.predict_file("field.jpg")  # or model.predict(rgb_array)
    print(json.dumps(model.to_json(plants), indent=1))

README.md describes the images the model expects and every output field. This file needs
torch, torchvision, timm, safetensors, numpy and opencv (requirements.txt), and nothing from
the training code: it repeats exactly the preprocessing the model was trained and evaluated
with.
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator

OTHER = "other_plant"  # a plant of a species the classifier doesn't know


class WeedModel:
    """The two models of this folder (model.json, detector.safetensors,
    classifier.safetensors), ready to predict.

    `device`: "cuda", "cpu", ... (default: CUDA when available). `amp`: half precision on
    CUDA, as the model was evaluated; faster, and the results differ only slightly.
    """

    def __init__(self, folder: str | Path | None = None, device: str | None = None,
                 amp: bool = True):
        folder = Path(folder) if folder is not None else Path(__file__).resolve().parent
        self.info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = amp and self.device.type == "cuda"
        self.detector = build_detector(self.info["detector"])
        self.detector.load_state_dict(load_file(folder / "detector.safetensors"))
        self.detector.to(self.device).eval()
        self.classes = {c["code"]: c for c in self.info["classes"]}
        self.codes = list(self.classes)
        self.classifier = build_classifier(self.info["classifier"], len(self.codes))
        self.classifier.load_state_dict(load_file(folder / "classifier.safetensors"))
        self.classifier.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray, min_plant_score: float | None = None,
                masks: bool = True) -> list[dict[str, Any]]:
        """The plants in an RGB image (uint8 array, height x width x 3), most certain first.

        `min_plant_score`: plants with a lower plant score are left out (default:
        handoff.min_plant_score of model.json). `masks`: also return each plant's mask as a
        boolean image (key "mask"); its outline is always in "polygons".
        """
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected an RGB uint8 image (height, width, 3), not "
                             f"{image.dtype} {image.shape}")
        if min_plant_score is None:
            min_plant_score = self.info["handoff"]["min_plant_score"]
        # mul(1 / 255), not div(255): bit for bit the scaling of training
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().mul(1 / 255).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            output = self.detector([tensor])[0]
        keep = output["scores"] >= min_plant_score
        if not keep.any():
            return []
        scores = output["scores"][keep].float().cpu().numpy()
        boxes = [[x0, y0, x1 - x0, y1 - y0]
                 for x0, y0, x1, y1 in output["boxes"][keep].float().cpu().numpy().tolist()]
        plant_masks = (output["masks"][keep, 0]
                       > self.info["detector"]["mask_threshold"]).cpu().numpy()

        settings = self.info["classifier"]
        inputs = torch.stack([crop_input(image, box, settings) for box in boxes])
        probs = []
        for batch in inputs.split(settings["batch_size"]):
            with torch.autocast(self.device.type, enabled=self.amp):
                logits = self.classifier(batch.to(self.device))
            probs.append(logits.float().softmax(1).cpu().numpy())
        probs = np.concatenate(probs)
        return [self._plant(box, float(score), mask, p, masks)
                for box, score, mask, p in zip(boxes, scores, plant_masks, probs)]

    def predict_file(self, path: str | Path, **options: Any) -> list[dict[str, Any]]:
        """`predict` on an image file (any format OpenCV reads)."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Can't read the image {path}")
        return self.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), **options)

    @staticmethod
    def to_json(plants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The plants without their mask images, ready for json.dumps."""
        return [{k: v for k, v in plant.items() if k != "mask"} for plant in plants]

    def _plant(self, box: list[float], plant_score: float, mask: np.ndarray, probs: np.ndarray,
               with_mask: bool) -> dict[str, Any]:
        handoff = self.info["handoff"]
        order = np.argsort(-probs)
        code, probability = self.codes[order[0]], float(probs[order[0]])
        reasons = []
        if code == OTHER:
            reasons.append("other_plant")
        elif probability < handoff["zero_shot_below_species_probability"]:
            reasons.append("low_species_probability")
        if plant_score < handoff["zero_shot_below_plant_score"]:
            reasons.append("low_plant_score")
        plant = {
            "box": [round(v, 1) for v in box],
            "polygons": polygons(mask),
            "plant_score": round(plant_score, 4),
            "species": code,
            "species_name": self.classes[code]["name"],
            "common_names": self.classes[code]["common_names"],
            "species_probability": round(probability, 4),
            "top5": [{"species": self.codes[k], "probability": round(float(probs[k]), 4)}
                     for k in order[:5]],
            # every class, in model.json's order; 4 significant digits rather than decimals,
            # so the small probabilities aren't rounded to 0
            "probabilities": {code: float(f"{p:.4g}")
                              for code, p in zip(self.codes, probs.tolist())},
            "send_to_zero_shot": bool(reasons),
            "zero_shot_reasons": reasons,
        }
        if with_mask:
            plant["mask"] = mask
        return plant


def build_detector(settings: dict[str, Any]) -> MaskRCNN:
    """The plant detector (Mask R-CNN, one class), without weights."""
    model = maskrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=2,  # background and plant
        min_size=(settings["min_size"],), max_size=settings["max_size"],
        box_detections_per_img=settings["detections_per_image"],
        box_score_thresh=settings["score_threshold"])
    sizes = settings["anchor_sizes"]
    model.rpn.anchor_generator = AnchorGenerator(tuple((size,) for size in sizes),
                                                 ((0.5, 1.0, 2.0),) * len(sizes))
    return model


def build_classifier(settings: dict[str, Any], num_classes: int) -> nn.Module:
    """The species classifier (timm), without weights; nothing is downloaded."""
    return timm.create_model(settings["backbone"], pretrained=False, num_classes=num_classes,
                             img_size=settings["input_size"])


def crop_input(image: np.ndarray, box: list[float], settings: dict[str, Any]) -> torch.Tensor:
    """The classifier's input for a plant: a square around its box (x, y, width, height),
    the box's longer side plus `margin` times it on each side, resized to `input_size`;
    normalised with `mean` and `std`."""
    x, y, w, h = box
    side = max(w, h) * (1 + 2 * settings["margin"])
    crop = cut_square(image, x + w / 2, y + h / 2, side, settings["input_size"],
                      tuple(settings["fill_rgb"]))
    tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
    mean = torch.tensor(settings["mean"]).view(3, 1, 1)
    std = torch.tensor(settings["std"]).view(3, 1, 1)
    return (tensor.float() / 255 - mean) / std


def cut_square(image: np.ndarray, cx: float, cy: float, side: float, size: int,
               fill: tuple[int, ...]) -> np.ndarray:
    """The square window (centre, side in px) of `image`, resized to size x size; parts
    outside the image get `fill`."""
    side_px = max(1, round(side))
    x0, y0 = round(cx - side_px / 2), round(cy - side_px / 2)
    x1, y1 = x0 + side_px, y0 + side_px
    height, width = image.shape[:2]
    patch = image[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    pad = (max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width))
    if any(pad):
        patch = cv2.copyMakeBorder(patch, *pad, cv2.BORDER_CONSTANT, value=fill)
    if patch.shape[:2] != (size, size):
        interpolation = cv2.INTER_AREA if side_px > size else cv2.INTER_LINEAR
        patch = cv2.resize(patch, (size, size), interpolation=interpolation)
    return patch


def polygons(mask: np.ndarray) -> list[list[list[int]]]:
    """Outer outlines of a mask as polygons [[x, y], ...] in pixels (a plant split by the
    image edge or a leaf gap can have several)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2).tolist() for contour in contours if len(contour) >= 3]
