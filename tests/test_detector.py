import cv2
import numpy as np
import pytest
import torch

from weedseg.config import load_config
from weedseg.detector.data import IGNORE_LABEL, CocoDetectionDataset, build_augmentation, collate
from weedseg.detector.engine import EpochSampler, predict, warmup_cosine
from weedseg.detector.model import build_model, model_settings
from weedseg.detector.run import (
    eval_dir,
    evaluated_splits,
    is_crash,
    resolve_run_dir,
    run_name,
    scale_dir,
    shared_run_dirs,
    split_file,
)
from weedseg.paths import project_root

H, W = 96, 128
MODEL_CONFIG = {"pretrained": False, "min_size": [96], "max_size": 128,
                "anchor_sizes": [16, 32, 64, 128, 256], "detections_per_image": 50,
                "score_threshold": 0.0}


@pytest.fixture
def coco(tmp_path):
    image = np.full((H, W, 3), 90, np.uint8)
    image[10:40, 20:50] = (40, 160, 40)
    cv2.imwrite(str(tmp_path / "a.jpg"), image)
    return {
        "info": {"image_dir": str(tmp_path)},
        "images": [{"id": 7, "file_name": "a.jpg", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "plant"}, {"id": 2, "name": "ignore"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 1, "bbox": [20, 10, 30, 30], "iscrowd": 0},
            {"id": 2, "image_id": 7, "category_id": 2, "bbox": [80, 60, 40, 30], "iscrowd": 1},
            {"id": 3, "image_id": 7, "category_id": 1, "bbox": [5, 5, 0.5, 9], "iscrowd": 0},
        ],
    }


def test_dataset_targets_label_ignore_regions_and_skip_degenerate_boxes(coco):
    image, target = CocoDetectionDataset(coco)[0]
    assert image.shape == (3, H, W) and image.dtype == torch.float32 and image.max() <= 1
    assert target["labels"].tolist() == [1, IGNORE_LABEL]
    assert target["boxes"].tolist() == [[20, 10, 50, 40], [80, 60, 120, 90]]
    assert "masks" not in target and target["image_id"] == 7


def test_flip_moves_the_boxes(coco):
    dataset = CocoDetectionDataset(coco, build_augmentation({"horizontal_flip": 1.0}))
    _, target = dataset[0]
    x0, _, x1, _ = target["boxes"][0].tolist()
    assert (x0, x1) == (W - 50, W - 20)


def test_rpn_ignores_anchors_on_ignore_regions():
    model = build_model(2, MODEL_CONFIG)
    target = {"boxes": torch.tensor([[0.0, 0, 32, 32], [64, 64, 96, 96]]),
              "labels": torch.tensor([1, IGNORE_LABEL])}
    anchors = torch.tensor([[0.0, 0, 32, 32], [64, 64, 96, 96], [200, 200, 230, 230]])
    labels, _ = model.rpn.assign_targets_to_anchors([anchors], [target])
    assert labels[0].tolist() == [1.0, -1.0, 0.0]


def test_model_has_no_mask_head():
    model = build_model(2, MODEL_CONFIG)
    assert not model.roi_heads.has_mask()
    assert model.roi_heads.box_predictor.cls_score.out_features == 2


def test_train_step_and_predict_run(coco):
    torch.manual_seed(0)
    model = build_model(2, MODEL_CONFIG)
    images, targets = collate([CocoDetectionDataset(coco)[0]])
    model.train()
    losses = model(images, targets)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()

    loader = torch.utils.data.DataLoader(CocoDetectionDataset(coco, with_targets=False),
                                         batch_size=1, collate_fn=collate)
    results = predict(model, loader, coco["categories"][:1], torch.device("cpu"), amp=False)
    assert results
    assert all(r["image_id"] == 7 and r["category_id"] == 1 for r in results)
    assert all(len(r["bbox"]) == 4 and 0 <= r["score"] <= 1 for r in results)
    assert all(set(r) == {"image_id", "category_id", "bbox", "score"} for r in results)


def test_epoch_sampler_resumes_mid_epoch():
    sampler = EpochSampler(10, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, skip=4)
    assert list(sampler) == full[4:] and len(sampler) == 6
    assert sorted(full) == list(range(10))


def test_warmup_cosine_schedule():
    factor = warmup_cosine(total_steps=100, warmup_steps=10)
    assert factor(0) == pytest.approx(0.001)
    assert factor(10) == pytest.approx(1.0)
    assert factor(55) == pytest.approx(0.5)
    assert factor(100) == pytest.approx(0.0)


def test_pretrained_weights_load_from_file(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "coco.pth"
    coco_model = build_model(91, MODEL_CONFIG)
    torch.save(coco_model.state_dict(), path)
    torch.manual_seed(1)
    model = build_model(2, {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(path)})
    assert torch.equal(model.backbone.body.conv1.weight, coco_model.backbone.body.conv1.weight)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2


def test_trainable_backbone_layers_hold_wherever_the_weights_come_from(tmp_path):
    path = tmp_path / "coco.pth"
    torch.save(build_model(91, MODEL_CONFIG).state_dict(), path)
    from_file = {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(path)}

    def trained(config):
        model = build_model(2, config)
        return {name.split(".")[0] for name, p in model.backbone.body.named_parameters()
                if p.requires_grad}

    every = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}
    assert trained({**from_file, "trainable_backbone_layers": 5}) == every
    # torchvision alone would train everything here (weights from a file count as none)
    assert trained({**from_file, "trainable_backbone_layers": 3}) == {"layer2", "layer3", "layer4"}
    assert trained({**MODEL_CONFIG, "trainable_backbone_layers": 0}) == set()
    model = build_model(2, {**from_file, "trainable_backbone_layers": 3})
    assert model.backbone.fpn.layer_blocks[0][0].weight.requires_grad  # FPN and heads always
    assert model.roi_heads.box_predictor.cls_score.weight.requires_grad
    with pytest.raises(ValueError, match="trainable_backbone_layers"):
        build_model(2, {**MODEL_CONFIG, "trainable_backbone_layers": 6})


def test_settings_from_before_trainable_layers_keep_what_torchvision_chose():
    # file weights and random init trained all 5; downloaded weights trained 3
    assert model_settings({"pretrained": True, "pretrained_path": "w.pth"})[
        "trainable_backbone_layers"] == 5
    assert model_settings({"pretrained": False, "pretrained_path": None})[
        "trainable_backbone_layers"] == 5
    assert model_settings({"pretrained": True, "pretrained_path": None})[
        "trainable_backbone_layers"] == 3
    # so a run saved before the setting resumes under today's config
    config = load_config("configs/detector.yaml")["model"]
    before = {k: v for k, v in config.items() if k != "trainable_backbone_layers"}
    assert model_settings(before) == model_settings(config)


def test_missing_pretrained_file_fails_clearly(tmp_path):
    config = {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(tmp_path / "none.pth")}
    with pytest.raises(FileNotFoundError, match="pretrained_path"):
        build_model(2, config)


def run_config(name):
    return {"run_name": name, "output_dir": "runs/detector",
            "data": {"dir": "data/processed/public5/plant_min20mm"}}


def test_run_name_placeholders_follow_the_data_folder():
    assert run_name(run_config("{dataset}_{taxonomy}")) == "public5_plant_min20mm"
    assert run_name(run_config("{taxonomy}_lr0.01")) == "plant_min20mm_lr0.01"
    assert run_name(run_config("plain_name")) == "plain_name"
    with pytest.raises(ValueError, match="species"):
        run_name(run_config("{species}"))


def test_run_dir_and_split_files_are_project_root_anchored():
    config = run_config("{dataset}")
    assert resolve_run_dir(config) == project_root() / "runs/detector/public5"
    assert split_file(config, "val") == (project_root() / "data/processed/public5"
                                         / "plant_min20mm/val.json")


def test_evaluated_splits_and_output_folders_of_a_run():
    config = {**run_config("{dataset}"),
              "evaluate": {"checkpoint": "best", "split": "val",
                           "extra_splits": ["test_a", "val"]},
              "evaluate_scale": {"split": "val"}}
    assert evaluated_splits(config) == ["val", "test_a"]
    run = project_root() / "runs/detector/public5"
    assert eval_dir(config) == run / "eval/val_best"
    assert eval_dir(config, "test_a") == run / "eval/test_a_best"
    assert scale_dir(config) == run / "scale/val_best"
    assert evaluated_splits({"evaluate": {"split": "test"}}) == ["test"]


def test_variants_need_run_folders_of_their_own():
    a, b = run_config("{dataset}"), run_config("{dataset}_scale")
    assert shared_run_dirs({"a.yaml": a, "b.yaml": b}) == []
    problem, = shared_run_dirs({"a.yaml": a, "b.yaml": b, "c.yaml": run_config("public5")})
    assert problem.startswith("a.yaml and c.yaml share the run folder")


def test_crashes_the_machine_caused_are_worth_a_restart_other_errors_not():
    # the error that ended the scale variant on the 4090 on 2026-11-11
    assert is_crash(RuntimeError("DataLoader worker (pid(s) 68844, 40408) exited unexpectedly"))
    assert is_crash(RuntimeError("CUDA error: unspecified launch failure"))
    assert is_crash(RuntimeError("[enforce fail at alloc_cpu.cpp:121] DefaultCPUAllocator: "
                                 "not enough memory: you tried to allocate 1073741824 bytes."))
    assert is_crash(OSError("[WinError 1455] The paging file is too small for this operation "
                            "to complete"))
    assert is_crash(MemoryError())
    assert not is_crash(FileNotFoundError("data/processed/public7/plant_min20mm/train.json"))
    assert not is_crash(KeyError("min_size"))
    # an error in the dataset code, raised again by PyTorch in the main process
    assert not is_crash(KeyError("Caught KeyError in DataLoader worker process 0.\n"
                                 "Original Traceback (most recent call last): ..."))


def test_scale_augmentation_trains_at_several_sizes_but_evaluates_at_the_last():
    model = build_model(2, {**MODEL_CONFIG, "min_size": [64, 80, 96]})
    assert model.transform.min_size == (64, 80, 96)
    model.eval()
    image = torch.rand(3, H, W)
    resized, _ = model.transform([image])
    assert resized.image_sizes[0] == (96, 128)
