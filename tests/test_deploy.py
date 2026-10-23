import ast

import numpy as np
import torch

from weedseg.classifier.crops import FILL_RGB
from weedseg.classifier.data import MEAN, STD
from weedseg.classifier.model import build_classifier
from weedseg.deploy import weed_model
from weedseg.deploy.export import describe_class, handoff_table
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.model import build_model
from weedseg.paths import resolve
from weedseg.twostage import detection_input

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "background": "keep",
         "dim_factor": 0.3}
SETTINGS = {"margin": 0.25, "input_size": 28, "fill_rgb": list(FILL_RGB),
            "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist(), "batch_size": 4,
            "backbone": "vit_small_patch14_dinov2.lvd142m"}
DETECTOR = {"min_size": 96, "max_size": 128, "anchor_sizes": [16, 32, 64, 128, 256],
            "detections_per_image": 50, "score_threshold": 0.05, "mask_threshold": 0.5}


def test_weed_model_needs_nothing_from_this_repo():
    tree = ast.parse(resolve("src/weedseg/deploy/weed_model.py").read_text(encoding="utf-8"))
    modules = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    assert modules <= {"json", "pathlib", "typing", "cv2", "numpy", "timm", "torch",
                       "safetensors", "torchvision"}


def test_crop_input_is_the_two_stage_prediction_input():
    image = np.random.default_rng(0).integers(0, 256, (60, 80, 3), np.uint8)
    for box in ([10.3, 12.7, 20.2, 9.9], [0, 0, 7, 30], [70.5, 50.2, 9.4, 9.7], [30, 25, 1, 1]):
        expected = detection_input(image, {"bbox": box}, CROPS)
        assert torch.equal(weed_model.crop_input(image, box, SETTINGS), expected)


def test_detector_is_the_trained_architecture():
    torch.manual_seed(0)
    trained = build_model(2, {**DETECTOR, "min_size": [DETECTOR["min_size"]],
                              "pretrained": False, "pretrained_path": None}).eval()
    deployed = weed_model.build_detector(DETECTOR).eval()
    deployed.load_state_dict(trained.state_dict())  # strict: same parameters and shapes
    image = torch.rand(3, 90, 120)
    with torch.no_grad():
        expected, result = trained([image])[0], deployed([image])[0]
    assert len(expected["boxes"]) > 0
    for key in ("boxes", "scores", "labels", "masks"):
        assert torch.equal(result[key], expected[key])


def test_classifier_is_the_trained_architecture():
    model = {"backbone": SETTINGS["backbone"], "pretrained": False, "pretrained_path": None,
             "freeze_blocks": 0, "drop_path": 0.1}
    trained = build_classifier(5, model, SETTINGS["input_size"]).eval()
    deployed = weed_model.build_classifier(SETTINGS, 5).eval()
    deployed.load_state_dict(trained.state_dict())
    inputs = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        assert torch.equal(deployed(inputs), trained(inputs))


def test_plants_go_to_zero_shot_for_each_reason():
    model = weed_model.WeedModel.__new__(weed_model.WeedModel)
    model.info = {"handoff": {"zero_shot_below_plant_score": 0.8,
                              "zero_shot_below_species_probability": 0.7}}
    model.codes = ["CHEAL", "ZEAMX", "other_plant"]
    model.classes = {code: {"name": code.lower(), "common_names": []} for code in model.codes}
    mask = np.zeros((10, 10), bool)
    mask[2:6, 3:8] = True

    def plant(score, probs):
        return model._plant([3, 2, 5, 4], score, mask, np.array(probs), with_mask=False)

    sure = plant(0.95, [0.85, 0.1, 0.05])
    assert (sure["species"], sure["send_to_zero_shot"], sure["zero_shot_reasons"]) == (
        "CHEAL", False, [])
    assert sure["top5"][0] == {"species": "CHEAL", "probability": 0.85}
    assert sure["polygons"] and "mask" not in sure
    assert plant(0.95, [0.6, 0.3, 0.1])["zero_shot_reasons"] == ["low_species_probability"]
    assert plant(0.95, [0.1, 0.1, 0.8])["zero_shot_reasons"] == ["other_plant"]
    assert plant(0.6, [0.2, 0.1, 0.7])["zero_shot_reasons"] == ["other_plant", "low_plant_score"]


def test_class_descriptions_and_handoff_table(tmp_path):
    taxa = load_taxa(resolve("configs/eppo_taxa.csv"))
    cheal = describe_class("CHEAL", taxa)
    assert (cheal["name"], cheal["family"]) == ("Chenopodium album", "Amaranthaceae")
    assert "fat-hen" in cheal["common_names"]
    assert describe_class("CENCY", taxa)["common_names"][0] == "bachelor's button"  # 2 spaces
    assert describe_class("other_plant", taxa)["common_names"] == []
    folder = tmp_path / "eval"
    folder.mkdir()
    (folder / "predictions.csv").write_text(
        "true,predicted,probability\nA,A,0.9\nA,B,0.6\nB,B,0.75\nB,other_plant,0.9\n",
        encoding="utf-8")
    table = handoff_table(folder, [0.5, 0.7, 0.95]).splitlines()
    assert table[2:] == ["| 0.5 | 100.0% | 66.7% |", "| 0.7 | 66.7% | 100.0% |",
                         "| 0.95 | 0% | - |"]
