import cv2
import numpy as np
import pytest
import torch

from weedseg.data.coco import encode_mask
from weedseg.maskrcnn.data import IGNORE_LABEL, CocoInstanceDataset, build_augmentation, collate
from weedseg.maskrcnn.engine import predict, warmup_cosine
from weedseg.maskrcnn.model import build_model

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


def test_missing_pretrained_file_fails_clearly(tmp_path):
    config = {**MODEL_CONFIG, "pretrained": True, "pretrained_path": str(tmp_path / "none.pth")}
    with pytest.raises(FileNotFoundError, match="pretrained_path"):
        build_model(3, config)
