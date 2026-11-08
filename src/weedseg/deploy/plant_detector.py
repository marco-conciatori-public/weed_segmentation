"""Plant detection in top-down field images: a box around every plant, without species.

A Faster R-CNN (torchvision, ResNet-50 FPN v2) with a single class, plant.

    from plant_detector import PlantDetector

    model = PlantDetector()                   # this folder; the GPU when there is one
    plants = model.predict_file("field.jpg")  # or model.predict(rgb_array)
    print(json.dumps(plants, indent=1))

README.md describes the images the model expects and the output. This file needs torch,
torchvision, safetensors, numpy and opencv (requirements.txt), and nothing from the training
code: it repeats exactly the preprocessing the model was trained and evaluated with.
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator


class PlantDetector:
    """The detector of this folder (model.json, detector.safetensors), ready to predict.

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

    @torch.no_grad()
    def predict(self, image: np.ndarray,
                min_plant_score: float | None = None) -> list[dict[str, Any]]:
        """The plants in an RGB image (uint8 array, height x width x 3), most certain first:
        {"box": [x, y, width, height] in pixels, "plant_score": 0 to 1}.

        `min_plant_score`: plants with a lower plant score are left out (default:
        min_plant_score of model.json).
        """
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Expected an RGB uint8 image (height, width, 3), not "
                             f"{image.dtype} {image.shape}")
        if min_plant_score is None:
            min_plant_score = self.info["min_plant_score"]
        # mul(1 / 255), not div(255): bit for bit the scaling of training
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().mul(1 / 255).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            output = self.detector([tensor])[0]
        keep = output["scores"] >= min_plant_score
        scores = output["scores"][keep].float().cpu().numpy().tolist()
        boxes = output["boxes"][keep].float().cpu().numpy().tolist()
        return [{"box": [round(v, 1) for v in (x0, y0, x1 - x0, y1 - y0)],
                 "plant_score": round(score, 4)}
                for (x0, y0, x1, y1), score in zip(boxes, scores)]

    def predict_file(self, path: str | Path, **options: Any) -> list[dict[str, Any]]:
        """`predict` on an image file (any format OpenCV reads)."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Can't read the image {path}")
        return self.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), **options)


def build_detector(settings: dict[str, Any]) -> FasterRCNN:
    """The plant detector (Faster R-CNN, one class), without weights."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=2,  # background and plant
        min_size=(settings["min_size"],), max_size=settings["max_size"],
        box_detections_per_img=settings["detections_per_image"],
        box_score_thresh=settings["score_threshold"])
    sizes = settings["anchor_sizes"]
    model.rpn.anchor_generator = AnchorGenerator(tuple((size,) for size in sizes),
                                                 ((0.5, 1.0, 2.0),) * len(sizes))
    return model
