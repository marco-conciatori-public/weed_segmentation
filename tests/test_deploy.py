import ast
import csv
import hashlib
import json
import shutil

import cv2
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from weedseg.classifier.crops import FILL_RGB
from weedseg.classifier.data import MEAN, STD
from weedseg.classifier.model import build_classifier
from weedseg.deploy import weed_model
from weedseg.data.coco import BOX_ONLY
from weedseg.deploy.export import (
    backbone_family,
    dataset_table,
    describe_class,
    handoff_table,
    split_table,
    training_data,
    write_splits,
)
from weedseg.deploy.inference import image_files, run_inference
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


@pytest.mark.parametrize("backbone, size", [("vit_small_patch14_dinov2.lvd142m", 28),
                                            ("vit_small_patch16_dinov3.lvd1689m", 32)])
def test_classifier_is_the_trained_architecture(backbone, size):
    model = {"backbone": backbone, "pretrained": False, "pretrained_path": None,
             "freeze_blocks": 0, "drop_path": 0.1}
    settings = {**SETTINGS, "backbone": backbone, "input_size": size}
    trained = build_classifier(5, model, size).eval()
    deployed = weed_model.build_classifier(settings, 5).eval()
    deployed.load_state_dict(trained.state_dict())
    inputs = torch.randn(2, 3, size, size)
    with torch.no_grad():
        assert torch.equal(deployed(inputs), trained(inputs))


def test_each_backbone_ships_its_licence():
    dinov3 = backbone_family("vit_base_patch16_dinov3.lvd1689m")
    licence = resolve(f"src/weedseg/deploy/{dinov3['file']}").read_text(encoding="utf-8")
    assert "provide a copy of this Agreement" in licence and "Built with DINOv3" in licence
    assert "Built with DINOv3" in dinov3["terms"]
    dinov2 = backbone_family("vit_base_patch14_dinov2.lvd142m")
    assert (dinov2["licence"], dinov2["file"], dinov2["terms"]) == ("Apache-2.0", None, "")
    with pytest.raises(SystemExit, match="No licence known"):
        backbone_family("resnet50.a1_in1k")


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
    assert sure["probabilities"] == {"CHEAL": 0.85, "ZEAMX": 0.1, "other_plant": 0.05}
    assert plant(0.95, [0.9, 0.1 - 1.23456e-7, 1.23456e-7])["probabilities"]["other_plant"] == (
        1.235e-7)  # significant digits: not rounded to 0
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


def test_training_data_says_what_each_dataset_trains():
    # a: masks, both stages; b: boxes only, which the detector leaves out; c: detector only.
    # The detector's older data lacks the names.
    a = {"source": "https://a.org (a copy)", "licence": "CC BY 4.0; see docs/datasets/a.md"}
    b = {"source": "https://b.org", "licence": "CC BY-NC 4.0"}
    images = {"a": {"file_name": "a/1.jpg", "dataset": "a"},
              "b": {"file_name": "b/1.jpg", "dataset": "b", BOX_ONLY: True},
              "c": {"file_name": "c/1.jpg", "dataset": "c"}}
    detector = {"info": {"datasets": {"a": a, "b": b, "c": {**b, "licence": "MIT"}}},
                "images": list(images.values())}
    classifier = {"info": {"datasets": {"a": {**a, "name": "A"}, "b": {**b, "name": "B"}}},
                  "images": [images["a"], images["b"]]}

    data = training_data(detector, classifier)
    assert data == {
        "a": {"name": "A", "source": a["source"], "licence": "CC BY 4.0",
              "trains": ["detector", "classifier"]},
        "b": {"name": "B", "source": b["source"], "licence": "CC BY-NC 4.0",
              "trains": ["classifier"]},
        "c": {"name": "c", "source": b["source"], "licence": "MIT", "trains": ["detector"]}}
    rows = dataset_table(detector["images"] + classifier["images"], data).splitlines()[2:]
    assert [[cell.strip() for cell in row.split("|")[1:-1]][1:5] for row in rows] == [
        ["CC BY 4.0", "unknown", "detector and classifier", "1"],
        ["CC BY-NC 4.0", "unknown", "classifier (boxes only)", "1"],
        ["MIT", "unknown", "detector", "1"]]


def test_splits_list_each_image_once_with_both_stages(tmp_path):
    raw = tmp_path / "raw" / "a"
    (raw / "sub").mkdir(parents=True)
    (raw / "sub" / "x.jpg").write_bytes(b"x")  # y.jpg and z.jpg are missing
    # z.jpg is labelled with boxes only: in the detector's data too, but not trained on
    images = {"train": ["sub/x.jpg", "z.jpg"], "val": ["y.jpg"], "test": []}
    for stage, stage_images in (("detector", {**images, "val": []}), ("classifier", images)):
        folder = tmp_path / stage
        folder.mkdir()
        for split, names in stage_images.items():
            coco = {"info": {"datasets": {"a": {"image_dir": f"{raw.as_posix()}/"}}},
                    "images": [{"file_name": f"{raw.as_posix()}/{name}", "dataset": "a",
                                "group": name[-5], **({BOX_ONLY: True} if name == "z.jpg" else {})}
                               for name in names]}
            (folder / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    rows = write_splits(tmp_path / "splits.csv", {s: tmp_path / s for s in ("detector", "classifier")})
    assert rows == [
        {"dataset": "a", "image": "sub/x.jpg", "sha256": hashlib.sha256(b"x").hexdigest(),
         "group": "x", "detector_split": "train", "classifier_split": "train"},
        {"dataset": "a", "image": "y.jpg", "sha256": "", "group": "y", "detector_split": "",
         "classifier_split": "val"},
        {"dataset": "a", "image": "z.jpg", "sha256": "", "group": "z", "detector_split": "",
         "classifier_split": "train"}]
    with open(tmp_path / "splits.csv", newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == rows
    assert split_table(rows).splitlines()[2:] == ["| train | 1 | 2 |", "| val | 0 | 1 |",
                                                  "| test | 0 | 0 |"]


def test_image_files_takes_files_folders_and_lists(tmp_path):
    for name in ("a.jpg", "b.PNG", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.jpg").write_bytes(b"")
    assert [p.name for p in image_files(str(tmp_path))] == ["a.jpg", "b.PNG"]
    assert [p.name for p in image_files([str(tmp_path / "sub"), str(tmp_path / "b.PNG")])] == [
        "c.jpg", "b.PNG"]
    with pytest.raises(ValueError, match="overwrite"):
        image_files([str(tmp_path), str(tmp_path / "a.jpg")])
    with pytest.raises(FileNotFoundError):
        image_files(str(tmp_path / "missing.jpg"))


def test_inference_runs_a_package_on_an_image(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    shutil.copyfile(resolve("src/weedseg/deploy/weed_model.py"), package / "weed_model.py")
    codes = ["CHEAL", "ZEAMX", "other_plant"]
    info = {"detector": {**DETECTOR, "score_threshold": 0.0},
            "classifier": SETTINGS,
            "handoff": {"min_plant_score": 0.0, "zero_shot_below_plant_score": 0.8,
                        "zero_shot_below_species_probability": 0.7},
            "classes": [{"code": c, "name": c.lower(), "common_names": []} for c in codes]}
    (package / "model.json").write_text(json.dumps(info), encoding="utf-8")
    torch.manual_seed(0)
    save_file(weed_model.build_detector(DETECTOR).state_dict(), package / "detector.safetensors")
    save_file(weed_model.build_classifier(SETTINGS, len(codes)).state_dict(),
              package / "classifier.safetensors")
    image = np.random.default_rng(0).integers(0, 256, (90, 120, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "field.png"), image)

    run_inference({"model": str(package), "images": str(tmp_path / "field.png"),
                   "output_dir": str(tmp_path / "out" / "{model}"), "device": "cpu",
                   "amp": False, "min_plant_score": None, "open_results": False,
                   "draw": {"label": "code", "mask_alpha": 0.35}})
    drawn = cv2.imread(str(tmp_path / "out" / "package" / "field.jpg"))
    assert drawn.shape == image.shape
    record = json.loads((tmp_path / "out" / "package" / "field.json").read_text("utf-8"))
    assert (record["width"], record["height"], record["model"]) == (120, 90, "package")
    assert all(plant["species"] in codes and "mask" not in plant for plant in record["plants"])
