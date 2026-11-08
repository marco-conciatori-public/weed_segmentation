import ast
import csv
import hashlib
import json
import shutil

import cv2
import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import save_file

from weedseg.data.combine import CATEGORIES
from weedseg.deploy import plant_detector
from weedseg.deploy.export import (
    dataset_table,
    export_package,
    licence_text,
    split_table,
    training_data,
    write_splits,
)
from weedseg.deploy.inference import image_files, run_inference
from weedseg.detector.model import build_model
from weedseg.paths import resolve

DETECTOR = {"min_size": 96, "max_size": 128, "anchor_sizes": [16, 32, 64, 128, 256],
            "detections_per_image": 50, "score_threshold": 0.05}


def training_model(seed=0):
    torch.manual_seed(seed)
    return build_model(2, {**DETECTOR, "min_size": [DETECTOR["min_size"]], "pretrained": False,
                           "pretrained_path": None})


def test_plant_detector_needs_nothing_from_this_repo():
    tree = ast.parse(resolve("src/weedseg/deploy/plant_detector.py").read_text(encoding="utf-8"))
    modules = {alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)}
    assert modules <= {"json", "pathlib", "typing", "cv2", "numpy", "torch", "safetensors",
                       "torchvision"}


def test_detector_is_the_trained_architecture():
    trained = training_model().eval()
    deployed = plant_detector.build_detector(DETECTOR).eval()
    deployed.load_state_dict(trained.state_dict())  # strict: same parameters and shapes
    image = torch.rand(3, 90, 120)
    with torch.no_grad():
        expected, result = trained([image])[0], deployed([image])[0]
    assert len(expected["boxes"]) > 0
    for key in ("boxes", "scores", "labels"):
        assert torch.equal(result[key], expected[key])


def write_package(folder, model):
    folder.mkdir()
    shutil.copyfile(resolve("src/weedseg/deploy/plant_detector.py"), folder / "plant_detector.py")
    info = {"detector": {**DETECTOR, "score_threshold": 0.0}, "min_plant_score": 0.0}
    (folder / "model.json").write_text(json.dumps(info), encoding="utf-8")
    save_file(model.state_dict(), folder / "detector.safetensors")


def test_predict_gives_the_training_models_boxes(tmp_path):
    model = training_model()
    write_package(tmp_path / "package", model)
    detector = plant_detector.PlantDetector(tmp_path / "package", device="cpu")
    image = np.random.default_rng(0).integers(0, 256, (90, 120, 3), np.uint8)
    plants = detector.predict(image)
    model.roi_heads.score_thresh = 0.0
    with torch.no_grad():
        expected = model.eval()([torch.from_numpy(image).permute(2, 0, 1).float() / 255])[0]
    assert len(plants) == len(expected["boxes"]) > 0
    x0, y0, x1, y1 = expected["boxes"][0].tolist()
    assert plants[0]["box"] == pytest.approx([x0, y0, x1 - x0, y1 - y0], abs=0.05)
    assert plants[0]["plant_score"] == pytest.approx(expected["scores"][0].item(), abs=1e-4)
    assert detector.predict(image, min_plant_score=1.1) == []
    with pytest.raises(ValueError, match="RGB uint8"):
        detector.predict(image.astype(np.float32))


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
    write_package(tmp_path / "package", training_model())
    image = np.random.default_rng(0).integers(0, 256, (90, 120, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "field.png"), image)
    run_inference({"model": str(tmp_path / "package"), "images": str(tmp_path / "field.png"),
                   "output_dir": str(tmp_path / "out" / "{model}"), "device": "cpu",
                   "amp": False, "min_plant_score": None, "open_results": False,
                   "draw": {"scores": True}})
    drawn = cv2.imread(str(tmp_path / "out" / "package" / "field.jpg"))
    assert drawn.shape == image.shape
    record = json.loads((tmp_path / "out" / "package" / "field.json").read_text("utf-8"))
    assert (record["width"], record["height"], record["model"]) == (120, 90, "package")
    assert record["plants"] and all(set(p) == {"box", "plant_score"} for p in record["plants"])


def test_training_data_dataset_table_and_licence():
    a = {"name": "A", "source": "https://a.org (a copy)",
         "licence": "CC BY 4.0; see docs/datasets/a.md"}
    b = {"source": "https://b.org", "licence": "CC BY-NC 4.0"}
    train = {"info": {"datasets": {"a": a, "b": b}},
             "images": [{"id": 1, "file_name": "a/1.jpg", "dataset": "a", "mm_per_px": 0.4,
                         "mm_per_px_estimated": True},
                        {"id": 2, "file_name": "b/1.jpg", "dataset": "b"}],
             "annotations": [{"image_id": 1, "category_id": 1}, {"image_id": 1, "category_id": 2},
                             {"image_id": 2, "category_id": 1}]}
    data = training_data(train)
    assert data == {"a": {"name": "A", "source": a["source"], "licence": "CC BY 4.0"},
                    "b": {"name": "b", "source": b["source"], "licence": "CC BY-NC 4.0"}}
    rows = dataset_table(train, data).splitlines()[2:]
    assert [[cell.strip() for cell in row.split("|")[1:-1]][1:5] for row in rows] == [
        ["CC BY 4.0", "0.40 mm/px (estimated)", "1", "1"], ["CC BY-NC 4.0", "unknown", "1", "1"]]
    assert "non-commercial research" in licence_text(data) and "(b;" in licence_text(data)
    assert "non-commercial research" not in licence_text({"a": data["a"]})


def test_splits_list_each_image_with_its_split(tmp_path):
    raw = tmp_path / "raw" / "a"
    (raw / "sub").mkdir(parents=True)
    (raw / "sub" / "x.jpg").write_bytes(b"x")  # y.jpg is missing
    data = tmp_path / "data"
    data.mkdir()
    for split, names in {"train": ["sub/x.jpg"], "val": ["y.jpg"], "test": []}.items():
        coco = {"info": {"datasets": {"a": {"image_dir": f"{raw.as_posix()}/"}}},
                "images": [{"file_name": f"{raw.as_posix()}/{name}", "dataset": "a",
                            "group": name[-5]} for name in names]}
        (data / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    rows = write_splits(tmp_path / "splits.csv", data)
    assert rows == [
        {"dataset": "a", "image": "sub/x.jpg", "sha256": hashlib.sha256(b"x").hexdigest(),
         "group": "x", "detector_split": "train"},
        {"dataset": "a", "image": "y.jpg", "sha256": "", "group": "y", "detector_split": "val"}]
    with open(tmp_path / "splits.csv", newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == rows
    assert split_table(rows).splitlines()[2:] == ["| a | 1 | 1 | 0 |"]


def test_export_writes_a_complete_package(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "1.png"), np.zeros((90, 120, 3), np.uint8))
    data = tmp_path / "data" / "both" / "plant_min20mm"
    data.mkdir(parents=True)
    info = {"datasets": {"cropandweed": {"name": "CropAndWeed", "source": "https://c.org",
                                         "licence": "Non-commercial use only; see x"}},
            "min_plant_size_mm": 20}
    for split in ("train", "val", "test"):
        coco = {"info": info, "categories": CATEGORIES, "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 30, 30]}],
                "images": [{"id": 1, "file_name": f"{images.as_posix()}/1.png", "group": "g",
                            "dataset": "cropandweed", "mm_per_px": 0.41}] if split == "train"
                else []}
        (data / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    config = {"run_name": "run", "output_dir": str(tmp_path / "runs"),
              "data": {"dir": str(data)}, "model": {**DETECTOR, "min_size": [96]},
              "evaluate": {"checkpoint": "best", "split": "val"}}
    (tmp_path / "detector.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    run = tmp_path / "runs" / "run"
    (run / "eval" / "val_best").mkdir(parents=True)
    torch.save({"model": training_model().state_dict(), "categories": [{"name": "plant"}],
                "config": config, "epoch": 3}, run / "best.pt")
    (run / "eval" / "val_best" / "by_dataset.csv").write_text(
        "dataset,images,plants,AP,AP50,found,missed,detections_on_no_plant\n"
        "all,5,40,0.5,0.8,0.9,4,7\ncropandweed,5,40,0.5,0.8,0.9,4,7\n", encoding="utf-8")
    (run / "eval" / "val_best" / "by_species.csv").write_text(
        "eppo,name,plants,found,datasets\nCHEAL,Chenopodium album,30,0.95,cropandweed 30\n"
        "1POAG,Poa,25,0.8,cropandweed 25\nZEAMX,Zea mays,5,1.0,cropandweed 5\n",
        encoding="utf-8")

    out = export_package({"detector": str(tmp_path / "detector.yaml"), "checkpoint": "best",
                          "output_dir": str(tmp_path / "exports"), "name": "pd_{date}",
                          "overwrite": False, "eppo_taxa": "configs/eppo_taxa.csv",
                          "min_plant_score": 0.5})
    assert sorted(p.name for p in out.iterdir()) == [
        "README.md", "detector.safetensors", "model.json", "plant_detector.py",
        "requirements.txt", "splits.csv"]
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert model["detector"]["architecture"].startswith("torchvision fasterrcnn_resnet50_fpn_v2")
    assert (model["detector"]["min_size"], model["detector"]["epoch"]) == (96, 3)
    assert (model["min_plant_score"], model["min_plant_size_mm"]) == (0.5, 20)
    readme = (out / "README.md").read_text(encoding="utf-8")
    # the table the NeutraWeed platform reads, and the species above the minimum
    assert "| Dataset | Val images | Labelled plants | Plants found |" in readme
    assert "| CropAndWeed | 5 | 40 | 90.0% | 7 | 0.800 |" in readme
    assert "| `CHEAL` | *Chenopodium album* | 30 | 95.0% |" in readme
    assert "| `1POAG` | *Poa* | 25 | 80.0% |" in readme
    assert "ZEAMX" not in readme and "1 more label has fewer than 20 val plants (" in readme
    assert "non-commercial research" in readme and "$" not in readme  # every placeholder filled
    detector = plant_detector.PlantDetector(out, device="cpu")
    assert detector.predict(np.zeros((90, 120, 3), np.uint8), min_plant_score=0.0) is not None
    with pytest.raises(SystemExit, match="exists"):
        export_package({"detector": str(tmp_path / "detector.yaml"), "checkpoint": "best",
                        "output_dir": str(tmp_path / "exports"), "name": out.name,
                        "overwrite": False, "eppo_taxa": "configs/eppo_taxa.csv",
                        "min_plant_score": 0.5})
