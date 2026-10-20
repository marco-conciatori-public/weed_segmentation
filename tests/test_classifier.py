import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.crops import (
    FILL_RGB,
    apply_background,
    cache_dir,
    cut_square,
    read_index,
    square_window,
    write_cache,
)
from weedseg.classifier.data import CropDataset, jitter_mask, sample_weights
from weedseg.classifier.engine import accuracy_metrics, train_one_epoch
from weedseg.classifier.model import build_classifier, parameter_groups
from weedseg.data.coco import encode_mask
from weedseg.maskrcnn.engine import EpochSampler

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "cache_quality": 95,
         "background": "keep", "dim_factor": 0.3}
AUGMENT = {"flips": True, "rotate90": True, "color_jitter": {"brightness": 0.2},
           "box_scale": 0.2, "box_shift": 0.1, "mask_jitter_px": 2}
# A tiny ViT with DINOv2's layout (patch 14, register-free), randomly initialised.
TINY_MODEL = {"backbone": "vit_small_patch14_dinov2.lvd142m", "pretrained": False,
              "pretrained_path": None, "freeze_blocks": 0, "drop_path": 0.0}


def test_square_window_centres_on_the_box_with_margin():
    assert square_window([10, 20, 40, 20], 0.25) == (30, 30, 60)
    assert square_window([10, 20, 40, 20], 0.25, scale=2, shift=(0.5, 0)) == (50, 30, 120)


def test_cut_square_pads_outside_the_image_and_resizes():
    image = np.zeros((10, 10, 3), np.uint8)
    image[:5, :5] = 255
    crop = cut_square(image, 0, 0, 10, 20, FILL_RGB)  # centred on the corner
    assert crop.shape == (20, 20, 3)
    assert tuple(crop[0, 0]) == FILL_RGB  # outside the image
    assert tuple(crop[15, 15]) == (255, 255, 255)
    mask = cut_square(np.eye(4, dtype=np.uint8), 2, 2, 4, 8, nearest=True)
    assert set(np.unique(mask)) == {0, 1}


def test_background_keep_dim_blank():
    crop = np.full((2, 2, 3), 200, np.uint8)
    mask = np.array([[True, False], [False, False]])
    assert apply_background(crop, mask, "keep", 0.3) is crop
    assert apply_background(crop, mask, "dim", 0.3)[0, 1, 0] == 60
    blank = apply_background(crop, mask, "blank", 0.3)
    assert tuple(blank[1, 1]) == FILL_RGB and blank[0, 0, 0] == 200
    with pytest.raises(ValueError, match="background"):
        apply_background(crop, mask, "grey", 0.3)
    grown = jitter_mask(np.pad(np.ones((2, 2), bool), 3), 1)
    assert grown.sum() == 12  # a 2x2 square grown by an ellipse of radius 1


@pytest.fixture
def cache(tmp_path):
    image = np.full((100, 120, 3), 90, np.uint8)
    image[20:50, 30:60] = (40, 160, 40)
    cv2.imwrite(str(tmp_path / "a.jpg"), image)
    mask = np.zeros((100, 120), bool)
    mask[20:50, 30:60] = True
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 120, "height": 100,
                    "mm_per_px": 0.5}],
        "categories": [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "SETVI"},
                       {"id": 3, "name": "ignore"}],
        "annotations": [
            {"id": 5, "image_id": 1, "category_id": 1, "bbox": [30, 20, 30, 30], "iscrowd": 0,
             "segmentation": encode_mask(mask)},
            {"id": 6, "image_id": 1, "category_id": 2, "bbox": [100, 80, 15, 15], "iscrowd": 0},
            {"id": 7, "image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "iscrowd": 1},
        ],
    }
    folder = cache_dir(tmp_path, CROPS)
    index = write_cache(coco, tmp_path, folder / "train", CROPS)
    return folder, read_index(index)


def test_cache_holds_every_labelled_plant(cache):
    folder, rows = cache
    assert [(r["annotation_id"], r["category"], r["size_mm"]) for r in rows] == [
        (5, "ZEAMX", 15.0), (6, "SETVI", 7.5)]  # the ignore region is left out
    crop = cv2.imread(str(folder / "train" / "5.jpg"))
    assert crop.shape == (42, 42, 3)  # input_size x cache_context
    mask = cv2.imread(str(folder / "train" / "5.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[21, 21] == 255 and mask[0, 0] == 0  # plant in the centre
    assert cv2.imread(str(folder / "train" / "6.png"), cv2.IMREAD_GRAYSCALE)[21, 21] == 255


@pytest.mark.parametrize("background", ["keep", "dim"])
def test_crop_dataset_samples(cache, background):
    folder, rows = cache
    crops = {**CROPS, "background": background}
    for augment in (None, AUGMENT):
        dataset = CropDataset(rows, folder / "train", ["SETVI", "ZEAMX"], crops, augment)
        image, label = dataset[0]
        assert image.shape == (3, 28, 28) and image.dtype == torch.float32
        assert label == 1 and dataset.labels == [1, 0]
    with pytest.raises(ValueError, match="ZEAMX"):
        CropDataset(rows, folder / "train", ["SETVI"], CROPS)


def test_sample_weights_balance_classes():
    labels = [0, 0, 0, 0, 1]
    assert sample_weights(labels, "none") is None
    inverse = sample_weights(labels, "inverse")
    assert sum(inverse[:4]) == pytest.approx(inverse[4])  # both classes equally likely
    sqrt = sample_weights(labels, "sqrt")
    assert sum(sqrt[:4]) == pytest.approx(2 * sqrt[4])  # in proportion to sqrt(4) : sqrt(1)
    with pytest.raises(ValueError):
        sample_weights(labels, "equal")


def test_classifier_builds_freezes_and_groups_parameters():
    model = build_classifier(3, {**TINY_MODEL, "freeze_blocks": 2}, 28)
    assert model(torch.rand(2, 3, 28, 28)).shape == (2, 3)
    assert not any(p.requires_grad for p in model.blocks[1].parameters())
    assert all(p.requires_grad for p in model.blocks[2].parameters())
    groups = parameter_groups(model, 1e-5, 1e-3, 0.05)
    head = [g for g in groups if g["lr"] == 1e-3]
    assert sum(p.numel() for g in head for p in g["params"]) == sum(
        p.numel() for p in model.get_classifier().parameters())
    assert all(g["weight_decay"] == 0.0 for g in groups if g["params"][0].ndim <= 1)


def test_accuracy_metrics():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = np.array([0, 0, 1, 1])
    metrics = accuracy_metrics(probs, labels, 2)
    assert metrics == {"accuracy": 0.75, "balanced_accuracy": 0.75}
    assert accuracy_metrics(probs[:3], labels[:3], 3)["balanced_accuracy"] == 1.0


def test_epoch_sampler_is_fixed_per_epoch_and_can_skip():
    sampler = EpochSampler(10, seed=0)
    sampler.set_epoch(3)
    order = list(sampler)
    assert sorted(order) == list(range(10))
    assert list(sampler) == order  # same epoch, same order
    sampler.set_epoch(3, skip=4)
    assert list(sampler) == order[4:] and len(sampler) == 6
    sampler.set_epoch(4)
    assert list(sampler) != order
    weighted = EpochSampler(10, seed=0, weights=[0] * 9 + [1])
    assert list(weighted) == [9] * 10


def test_train_one_epoch_resumes_mid_epoch():
    torch.manual_seed(0)
    data = torch.utils.data.TensorDataset(torch.randn(24, 5), torch.randint(0, 3, (24,)))
    model = torch.nn.Linear(5, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = EpochSampler(len(data), seed=0)
    loader = torch.utils.data.DataLoader(data, 4, sampler=sampler, drop_last=True)

    def epoch(**kwargs):
        return train_one_epoch(model, loader, optimizer, scheduler, scaler,
                               torch.nn.CrossEntropyLoss(), torch.device("cpu"), False, None, 0,
                               **kwargs)

    saved = []
    sampler.set_epoch(0)
    epoch(save_every=4, save=saved.append)
    assert [s["batches"] for s in saved] == [4] and saved[0]["seen"] == 16
    sampler.set_epoch(0, skip=saved[0]["batches"] * 4)  # crash after batch 4: the last 2 batches
    result = epoch(partial=saved[0])
    assert 0 <= result["accuracy"] <= 1
    assert len(loader) == 2
