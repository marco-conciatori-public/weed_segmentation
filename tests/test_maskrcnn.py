import cv2
import numpy as np
import pytest
import torch
from torchvision.models.detection.roi_heads import RoIHeads

from weedseg.config import load_config
from weedseg.data.coco import encode_mask
from weedseg.maskrcnn.data import IGNORE_LABEL, CocoInstanceDataset, build_augmentation, collate
from weedseg.maskrcnn.engine import predict, warmup_cosine
from weedseg.maskrcnn.model import build_model, model_settings
from weedseg.maskrcnn.run import resolve_run_dir, run_name, split_file
from weedseg.paths import project_root

H, W = 96, 128
MODEL_CONFIG = {"pretrained": False, "min_size": [96], "max_size": 128,
                "anchor_sizes": [16, 32, 64, 128, 256], "detections_per_image": 50,
                "score_threshold": 0.0, "mask_threshold": 0.5}


@pytest.fixture
def coco(tmp_path):
    image = np.full((H, W, 3), 90, np.uint8)
    image[10:40, 20:50] = (40, 160, 40)
    cv2.imwrite(str(tmp_path / "a.jpg"), image)
    mask = np.zeros((H, W), bool)
    mask[10:40, 20:50] = True
    region = np.zeros((H, W), bool)
    region[60:90, 80:120] = True
    return {
        "info": {"image_dir": str(tmp_path)},
        "images": [{"id": 7, "file_name": "a.jpg", "width": W, "height": H}],
        "categories": [{"id": 1, "name": "crop"}, {"id": 2, "name": "weed"},
                       {"id": 3, "name": "ignore"}],
        "annotations": [
            {"id": 1, "image_id": 7, "category_id": 2, "bbox": [20, 10, 30, 30], "iscrowd": 0,
             "segmentation": encode_mask(mask)},
            {"id": 2, "image_id": 7, "category_id": 3, "bbox": [80, 60, 40, 30], "iscrowd": 1,
             "segmentation": encode_mask(region)},
        ],
    }


def test_dataset_targets_label_ignore_regions(coco):
    image, target = CocoInstanceDataset(coco)[0]
    assert image.shape == (3, H, W) and image.dtype == torch.float32 and image.max() <= 1
    assert target["labels"].tolist() == [2, IGNORE_LABEL]  # weed = second category
    assert target["boxes"].tolist()[0] == [20, 10, 50, 40]
    assert target["masks"].shape == (2, H, W) and target["masks"][0].sum() == 900
    assert target["image_id"] == 7


def test_flip_moves_boxes_and_masks_together(coco):
    dataset = CocoInstanceDataset(coco, build_augmentation({"horizontal_flip": 1.0}))
    _, target = dataset[0]
    x0, _, x1, _ = target["boxes"][0].tolist()
    assert (x0, x1) == (W - 50, W - 20)
    columns = target["masks"][0].any(dim=0).nonzero().flatten()
    assert (columns.min().item(), columns.max().item() + 1) == (W - 50, W - 20)


def test_rpn_ignores_anchors_on_ignore_regions():
    model = build_model(3, MODEL_CONFIG)
    target = {"boxes": torch.tensor([[0.0, 0, 32, 32], [64, 64, 96, 96]]),
              "labels": torch.tensor([1, IGNORE_LABEL])}
    anchors = torch.tensor([[0.0, 0, 32, 32], [64, 64, 96, 96], [200, 200, 230, 230]])
    labels, _ = model.rpn.assign_targets_to_anchors([anchors], [target])
    assert labels[0].tolist() == [1.0, -1.0, 0.0]


def test_train_step_and_predict_run(coco):
    torch.manual_seed(0)
    model = build_model(3, MODEL_CONFIG)
    images, targets = collate([CocoInstanceDataset(coco)[0]])
    model.train()
    losses = model(images, targets)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()

    loader = torch.utils.data.DataLoader(CocoInstanceDataset(coco, with_targets=False),
                                         batch_size=1, collate_fn=collate)
    results = predict(model, loader, coco["categories"][:2], torch.device("cpu"), amp=False)
    assert all(r["image_id"] == 7 and r["category_id"] in (1, 2) for r in results)
    assert all(len(r["bbox"]) == 4 and "counts" in r["segmentation"] for r in results)
    for r in results:  # the detection's own class is among its class scores, with its score
        assert r["class_scores"][str(r["category_id"])] == pytest.approx(r["score"], abs=1e-4)
        assert set(r["class_scores"]) <= {"1", "2"} and sum(r["class_scores"].values()) <= 1.001


def test_class_score_heads_detect_like_torchvision():
    torch.manual_seed(0)
    model = build_model(3, MODEL_CONFIG).eval()
    images = [torch.rand(3, H, W)]
    with torch.no_grad():
        ours = model(images)[0]
        model.roi_heads.__class__ = RoIHeads
        stock = model(images)[0]
    assert len(ours["boxes"]) > 0
    for key in ("boxes", "scores", "labels"):
        assert torch.equal(ours[key], stock[key])


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
    model = build_model(3, {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(path)})
    assert torch.equal(model.backbone.body.conv1.weight, coco_model.backbone.body.conv1.weight)
    assert model.roi_heads.box_predictor.cls_score.out_features == 3


def test_trainable_backbone_layers_hold_wherever_the_weights_come_from(tmp_path):
    path = tmp_path / "coco.pth"
    torch.save(build_model(91, MODEL_CONFIG).state_dict(), path)
    from_file = {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(path)}

    def trained(config):
        model = build_model(3, config)
        return {name.split(".")[0] for name, p in model.backbone.body.named_parameters()
                if p.requires_grad}

    every = {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"}
    assert trained({**from_file, "trainable_backbone_layers": 5}) == every
    # torchvision alone would train everything here (weights from a file count as none)
    assert trained({**from_file, "trainable_backbone_layers": 3}) == {"layer2", "layer3", "layer4"}
    assert trained({**MODEL_CONFIG, "trainable_backbone_layers": 0}) == set()
    model = build_model(3, {**from_file, "trainable_backbone_layers": 3})
    assert model.backbone.fpn.layer_blocks[0][0].weight.requires_grad  # FPN and heads always
    assert model.roi_heads.mask_predictor.mask_fcn_logits.weight.requires_grad
    with pytest.raises(ValueError, match="trainable_backbone_layers"):
        build_model(3, {**MODEL_CONFIG, "trainable_backbone_layers": 6})


def test_settings_from_before_trainable_layers_keep_what_torchvision_chose():
    # file weights and random init trained all 5; downloaded weights trained 3
    assert model_settings({"pretrained": True, "pretrained_path": "w.pth"})[
        "trainable_backbone_layers"] == 5
    assert model_settings({"pretrained": False, "pretrained_path": None})[
        "trainable_backbone_layers"] == 5
    assert model_settings({"pretrained": True, "pretrained_path": None})[
        "trainable_backbone_layers"] == 3
    # so a run saved before the setting resumes under today's configs
    for path in ("configs/maskrcnn.yaml", "configs/detector.yaml"):
        config = load_config(path)["model"]
        before = {k: v for k, v in config.items() if k != "trainable_backbone_layers"}
        assert model_settings(before) == model_settings(config)


def test_missing_pretrained_file_fails_clearly(tmp_path):
    config = {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(tmp_path / "none.pth")}
    with pytest.raises(FileNotFoundError, match="pretrained_path"):
        build_model(3, config)


def run_config(name):
    return {"run_name": name, "output_dir": "runs/maskrcnn",
            "data": {"dir": "data/processed/cropandweed/species_min50img_min20mm"}}


def test_run_name_placeholders_follow_the_data_folder():
    assert (run_name(run_config("{dataset}_{taxonomy}"))
            == "cropandweed_species_min50img_min20mm")
    assert run_name(run_config("{taxonomy}_lr0.01")) == "species_min50img_min20mm_lr0.01"
    assert run_name(run_config("plain_name")) == "plain_name"
    with pytest.raises(ValueError, match="species"):
        run_name(run_config("{species}"))


def test_run_dir_and_split_files_are_project_root_anchored():
    config = run_config("{dataset}")
    assert resolve_run_dir(config) == project_root() / "runs/maskrcnn/cropandweed"
    assert split_file(config, "val") == (project_root() / "data/processed/cropandweed"
                                         / "species_min50img_min20mm/val.json")
