import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.model import build_classifier
from weedseg.config import load_config
from weedseg.data.coco import encode_mask
from weedseg.twostage import (
    classify_detections,
    config_problems,
    detection_input,
    shared_run_dirs,
    with_species,
)

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "background": "keep",
         "dim_factor": 0.3}
TINY_MODEL = {"backbone": "vit_small_patch14_dinov2.lvd142m", "pretrained": False,
              "pretrained_path": None, "freeze_blocks": 0, "drop_path": 0.0}


def test_species_scores_keep_plant_and_class_apart():
    detection = {"image_id": 1, "category_id": 1, "score": 0.8, "bbox": [0, 0, 5, 5],
                 "class_scores": {"1": 0.8}}
    result = with_species(detection, np.array([0.1, 0.6, 0.2999, 0.0001]), [11, 12, 13, 14])
    assert result["category_id"] == 12
    assert result["score"] == result["plant_score"] == 0.8
    assert result["class_scores"] == {"11": 0.1, "12": 0.6, "13": 0.2999}  # 14 too small
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
        assert r["score"] == r["plant_score"]
        assert sum(r["class_scores"].values()) == pytest.approx(1, abs=0.01)


def test_config_problems_catch_mismatched_data():
    species = "data/processed/x/species_min50img_min20mm"
    plant = {"source_dir": species}
    detector = {"data": {"dir": "data/processed/x/plant_min20mm"}}
    same = {"data": {"dir": species}}
    assert config_problems(plant, detector, same, same) == []
    problems = config_problems(plant, {"data": {"dir": species}}, same,
                               {"data": {"dir": "data/processed/y/species"}})
    assert len(problems) == 2
    assert "twostage" in problems[0] and "detector" in problems[1]


def test_shipped_configs_fit_together():
    twostage = load_config("configs/twostage.yaml")
    plant = load_config("configs/plant_dataset.yaml")
    detector = load_config(twostage["detector"]["config"])
    classifier_path = twostage["classifier"]["config"]
    classifiers = {classifier_path: load_config(classifier_path)}
    classifiers |= {path: load_config(path)
                    for path in load_config("configs/run_twostage.yaml")["classifier_variants"]}
    for classifier in classifiers.values():
        assert config_problems(plant, detector, classifier, twostage) == []
    assert shared_run_dirs(classifiers) == []


@pytest.mark.parametrize("path, changed", [
    ("configs/classifier_seed1.yaml", {"seed": 1}),
    ("configs/classifier_detail.yaml",
     {"augment.detail_p": 0.5, "augment.blur_p": 0.2, "augment.jpeg_p": 0.3}),
    ("configs/classifier_colour.yaml",
     {f"augment.color_jitter.{k}": v for k, v in
      {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.05}.items()}),
    ("configs/classifier_frozen.yaml", {"model.freeze_blocks": 12}),
    ("configs/classifier_dinov3.yaml", {"model.backbone": "vit_base_patch16_dinov3.lvd1689m"}),
    ("configs/classifier_balance.yaml", {"train.balance": "sqrt"}),
])
def test_variants_change_only_what_they_name(path, changed):
    def flat(config, prefix=""):
        items = {}
        for key, value in config.items():
            items |= (flat(value, f"{prefix}{key}.") if isinstance(value, dict)
                      else {f"{prefix}{key}": value})
        return items

    base, variant = flat(load_config("configs/classifier.yaml")), flat(load_config(path))
    differing = {key: variant.get(key) for key in base.keys() | variant.keys()
                 if base.get(key) != variant.get(key)}
    assert differing.pop("run_name") != base["run_name"]
    assert differing == changed


def test_shared_run_dirs():
    config = {"run_name": "{dataset}_a", "output_dir": "runs/x",
              "data": {"dir": "data/processed/d/species"}}
    assert shared_run_dirs({"a.yaml": config, "b.yaml": {**config, "run_name": "{dataset}_b"}}) == []
    (problem,) = shared_run_dirs({"a.yaml": config, "b.yaml": dict(config)})
    assert problem.startswith("a.yaml and b.yaml train the same run")
