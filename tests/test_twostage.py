import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.model import build_classifier
from weedseg.config import load_config
from weedseg.data.coco import encode_mask
from weedseg.paths import resolve
from weedseg.twostage import (
    classify_detections,
    config_problems,
    detection_input,
    missing_data,
    other_data,
    shared_run_dirs,
    with_species,
)

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "background": "keep",
         "dim_factor": 0.3}
TINY_MODEL = {"backbone": "vit_small_patch14_dinov2.lvd142m", "pretrained": False,
              "pretrained_path": None, "freeze_blocks": 0, "drop_path": 0.0}
# The DINOv2 variants, since the base classifier is DINOv3 (2026-10-28).
DINOV2 = {"model.backbone": "vit_base_patch14_dinov2.lvd142m"}


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
    runner = load_config("configs/run_twostage.yaml")
    classifiers |= {path: load_config(path) for path in runner["classifier_variants"]}
    # every classifier config, listed or not, trains a run of its own on the same data
    classifiers |= {f"configs/{path.name}": load_config(path)
                    for path in resolve("configs").glob("classifier*.yaml")}
    main = classifiers[classifier_path]
    combine = load_config("configs/combine.yaml")
    assert resolve(main["data"]["dir"]).parent.name == combine["name"]
    variant_data = {variant["name"] for variant in combine["variants"]}
    for classifier in classifiers.values():
        if other_data(main, classifier):  # a variant on another combination of combine.yaml
            data = resolve(classifier["data"]["dir"])
            assert data.parent.name in variant_data
            assert data.name == resolve(main["data"]["dir"]).name
        else:
            assert config_problems(plant, detector, classifier, twostage) == []
    assert shared_run_dirs(classifiers) == []
    # read only after hours of training, by the comparison at the end
    assert all(0 <= share < 1 for share in load_config(runner["compare_runs"])["crops_removed"])


@pytest.mark.parametrize("path, changed", [
    ("configs/classifier_dinov2.yaml", DINOV2),
    ("configs/classifier_seed1.yaml", {**DINOV2, "seed": 1}),
    ("configs/classifier_detail.yaml",
     {**DINOV2, "augment.detail_p": 0.5, "augment.blur_p": 0.2, "augment.jpeg_p": 0.3}),
    ("configs/classifier_colour.yaml",
     {**DINOV2, **{f"augment.color_jitter.{k}": v for k, v in
                   {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4,
                    "hue": 0.05}.items()}}),
    ("configs/classifier_frozen.yaml", {**DINOV2, "model.freeze_blocks": 12}),
    ("configs/classifier_balance.yaml", {**DINOV2, "train.balance": "sqrt"}),
    ("configs/classifier_seed2.yaml", {**DINOV2, "seed": 2}),
    ("configs/classifier_dinov3_seed1.yaml", {"seed": 1}),
    ("configs/classifier_dinov3_seed2.yaml", {"seed": 2}),
    ("configs/classifier_dinov3_frozen.yaml", {"model.freeze_blocks": 12}),
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


def flat_config(path):
    def flat(config, prefix=""):
        items = {}
        for key, value in config.items():
            items |= (flat(value, f"{prefix}{key}.") if isinstance(value, dict)
                      else {f"{prefix}{key}": value})
        return items
    return flat(load_config(path))


def test_imageweeds_variants_change_only_the_data_and_seed():
    base = flat_config("configs/classifier.yaml")
    variant = flat_config("configs/classifier_imageweeds.yaml")
    assert {k for k in base.keys() | variant.keys() if base.get(k) != variant.get(k)} == {
        "data.dir"}
    seed1 = flat_config("configs/classifier_imageweeds_seed1.yaml")
    assert {k: seed1[k] for k in seed1 if seed1[k] != variant.get(k)} == {
        "run_name": "{dataset}_{taxonomy}_dinov3b_seed1", "seed": 1}


def test_variants_on_other_data(tmp_path):
    main = {"data": {"dir": str(tmp_path / "public7" / "species")}}
    seed1 = {"data": {"dir": str(tmp_path / "public7" / "species")}}
    imageweeds = {"data": {"dir": str(tmp_path / "public7+imageweeds" / "species")}}
    assert not other_data(main, seed1) and other_data(main, imageweeds)
    (tmp_path / "public7" / "species").mkdir(parents=True)
    (tmp_path / "public7" / "species" / "train.json").write_text("{}")
    (problem,) = missing_data({"a.yaml": main, "b.yaml": imageweeds})
    assert problem.startswith("b.yaml: no ") and "combine_datasets.py" in problem
