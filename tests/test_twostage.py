import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.model import build_classifier
from weedseg.data.coco import encode_mask
from weedseg.twostage import classify_detections, detection_input, with_species

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "background": "keep",
         "dim_factor": 0.3}
TINY_MODEL = {"backbone": "vit_small_patch14_dinov2.lvd142m", "pretrained": False,
              "pretrained_path": None, "freeze_blocks": 0, "drop_path": 0.0}


def test_species_scores_are_plant_score_times_class_probability():
    detection = {"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [0, 0, 5, 5],
                 "class_scores": {"1": 0.8}}
    result = with_species(detection, np.array([0.1, 0.6, 0.2999, 0.0001]), [11, 12, 13, 14])
    assert result["category_id"] == 12
    assert result["score"] == pytest.approx(0.48)
    assert result["plant_score"] == 0.8
    assert result["class_scores"] == {"11": 0.08, "12": 0.48, "13": 0.2399}  # 14 too small
    assert result["bbox"] == [0, 0, 5, 5]


def test_detection_input_uses_the_mask_only_when_needed():
    image = np.full((60, 60, 3), 200, np.uint8)
    mask = np.zeros((60, 60), bool)
    mask[20:40, 20:40] = True
    detection = {"bbox": [20, 20, 20, 20], "segmentation": encode_mask(mask)}
    kept = detection_input(image, detection, CROPS)
    dimmed = detection_input(image, detection, {**CROPS, "background": "dim"})
    assert kept.shape == dimmed.shape == (3, 28, 28)
    assert torch.equal(kept[:, 14, 14], dimmed[:, 14, 14])  # the plant itself
    assert (dimmed[:, 1, 1] < kept[:, 1, 1]).all()  # its surroundings


def test_classify_detections_end_to_end(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), np.full((60, 80, 3), 120, np.uint8))
    images = [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "missing.jpg"}]
    detections = [{"image_id": 1, "category_id": 1, "score": s, "bbox": [10, 10, 20, 20]}
                  for s in (0.9, 0.5, 0.01)]
    torch.manual_seed(0)
    model = build_classifier(3, TINY_MODEL, 28)
    results = classify_detections(detections, images, tmp_path, model, [7, 8, 9], CROPS,
                                  torch.device("cpu"), amp=False, batch_size=2,
                                  min_plant_score=0.05)
    assert [r["plant_score"] for r in results] == [0.9, 0.5]  # 0.01 dropped, no 2nd image
    for r in results:
        assert r["category_id"] in (7, 8, 9)
        assert sum(r["class_scores"].values()) == pytest.approx(r["plant_score"], abs=0.01)
