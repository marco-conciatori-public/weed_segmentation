import copy
import csv
import random
from collections import Counter

import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.crops import (
    FILL_RGB,
    apply_background,
    cache_dir,
    cut_square,
    index_is_current,
    read_index,
    square_window,
    write_cache,
)
from weedseg.classifier.data import CropDataset, jitter_mask, lower_detail, sample_weights
from weedseg.classifier.engine import (
    PartialLabelLoss,
    accuracy_metrics,
    class_by_dataset,
    label_fit,
    last_below_best,
    train_one_epoch,
)
from weedseg.classifier.model import build_classifier, parameter_groups
from weedseg.data.coco import encode_mask
from weedseg.maskrcnn.engine import EpochSampler

CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "cache_quality": 95,
         "background": "keep", "dim_factor": 0.3}
AUGMENT = {"flips": True, "rotate90": True, "color_jitter": {"brightness": 0.2},
           "box_scale": 0.2, "box_shift": 0.1, "mask_jitter_px": 2,
           "detail_p": 1.0, "max_mm_per_px": 1.2, "blur_p": 0.5, "blur_sigma": [0.3, 1.0],
           "jpeg_p": 0.5, "jpeg_quality": [40, 90]}
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
            # a hidden plant with a partial label, e.g. a weed of unknown species
            {"id": 8, "image_id": 1, "category_id": 3, "bbox": [60, 60, 20, 20], "iscrowd": 1,
             "eppo": "1PLAK", "hidden_plant": True, "allowed_classes": ["SETVI", "other_plant"]},
        ],
    }
    folder = cache_dir(tmp_path, CROPS)
    index = write_cache(coco, tmp_path, folder / "train", CROPS)
    return folder, read_index(index), coco, index


def test_cache_holds_every_labelled_plant(cache):
    folder, rows, _, _ = cache
    assert [(r["annotation_id"], r["category"], r["size_mm"], r["allowed"]) for r in rows] == [
        (5, "ZEAMX", 15.0, []), (6, "SETVI", 7.5, []),
        (8, "ignore", 10.0, ["SETVI", "other_plant"])]  # the plain ignore region is left out
    crop = cv2.imread(str(folder / "train" / "5.jpg"))
    assert crop.shape == (42, 42, 3)  # input_size x cache_context
    mask = cv2.imread(str(folder / "train" / "5.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[21, 21] == 255 and mask[0, 0] == 0  # plant in the centre
    assert cv2.imread(str(folder / "train" / "6.png"), cv2.IMREAD_GRAYSCALE)[21, 21] == 255


def test_cache_index_made_from_other_data_is_not_current(cache):
    _, _, coco, index = cache
    assert index_is_current(index, coco)
    changed = copy.deepcopy(coco)
    changed["annotations"][3]["allowed_classes"] = ["SETVI"]
    assert not index_is_current(index, changed)
    changed["annotations"].pop()
    assert not index_is_current(index, changed)  # the plant is gone
    with open(index, newline="", encoding="utf-8") as f:
        old = [{k: v for k, v in row.items() if k != "allowed"} for row in csv.DictReader(f)]
    with open(index, "w", newline="", encoding="utf-8") as f:  # an index from before 2026-10-26
        writer = csv.DictWriter(f, list(old[0]))
        writer.writeheader()
        writer.writerows(old)
    assert not index_is_current(index, coco)
    assert read_index(index)[2]["allowed"] == []


@pytest.mark.parametrize("background", ["keep", "dim"])
def test_crop_dataset_samples(cache, background):
    folder, rows, _, _ = cache
    crops = {**CROPS, "background": background}
    classes = ["SETVI", "ZEAMX", "other_plant"]
    for augment in (None, AUGMENT):
        dataset = CropDataset(rows, folder / "train", classes, crops, augment)
        image, label, allowed = dataset[0]
        assert image.shape == (3, 28, 28) and image.dtype == torch.float32
        assert label == 1 and allowed.tolist() == [False, True, False]
        assert dataset.labels == [1, 0, -1]  # -1: a partial label
        assert dataset[2][2].tolist() == [True, False, True]
        assert dataset.balance_keys() == [1, 0, (0, 2)]
    with pytest.raises(ValueError, match="ZEAMX"):
        CropDataset(rows, folder / "train", ["SETVI", "other_plant"], CROPS)
    with pytest.raises(ValueError, match="other_plant"):
        CropDataset(rows, folder / "train", ["SETVI", "ZEAMX"], CROPS)


def test_crop_detail_is_the_coarser_of_source_cache_and_input(cache):
    folder, rows, _, _ = cache
    dataset = CropDataset(rows, folder / "train", ["SETVI", "ZEAMX", "other_plant"], CROPS)
    # Plant 5: 15 mm at 0.5 mm/px; its 22.5 mm crop has 45 source px but 28 at input_size.
    window_mm, mm_per_px = dataset.detail(0, 28.0, 1.0)
    assert window_mm == pytest.approx(22.5) and mm_per_px == pytest.approx(22.5 / 28)
    rows[0]["mm_per_px"] = 2.0  # a coarse source: its own detail counts
    assert dataset.detail(0, 28.0, 1.0)[1] == pytest.approx(2.0)
    rows[0]["mm_per_px"] = None
    assert dataset.detail(0, 28.0, 1.0) == (None, None)


def test_lower_detail():
    random.seed(0)
    crop = np.random.default_rng(0).integers(0, 256, (64, 64, 3), np.uint8)

    def sharpness(image):
        return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

    off = {"detail_p": 0.0, "blur_p": 0.0, "jpeg_p": 0.0}
    assert np.array_equal(lower_detail(crop, off, 30.0, 0.15), crop)
    assert np.array_equal(lower_detail(crop, {}, 30.0, 0.15), crop)  # configs from before
    detail = {**off, "detail_p": 1.0, "max_mm_per_px": 1.2}
    coarser = lower_detail(crop, detail, 30.0, 1.0)  # to 25-30 px of detail
    assert coarser.shape == crop.shape and coarser.dtype == np.uint8
    assert sharpness(coarser) < sharpness(crop) / 2
    assert np.array_equal(lower_detail(crop, detail, 30.0, 1.5), crop)  # already coarser
    # 64 px across 30 mm hold 0.47 mm/px at best, whatever the source's
    assert np.array_equal(lower_detail(crop, {**detail, "max_mm_per_px": 0.4}, 30.0, 0.15),
                          crop)
    assert np.array_equal(lower_detail(crop, detail, None, None), crop)  # resolution unknown
    blurred = lower_detail(crop, {**off, "blur_p": 1.0, "blur_sigma": [1.0, 1.0]}, None, None)
    assert sharpness(blurred) < sharpness(crop) / 2
    compressed = lower_detail(crop[:, ::-1], {**off, "jpeg_p": 1.0, "jpeg_quality": [40, 40]},
                              None, None)  # a flipped view, as after flip_rotate
    assert compressed.shape == crop.shape and not np.array_equal(compressed, crop[:, ::-1])


def test_accuracy_by_class_and_dataset():
    true = ["CHEAL"] * 4 + ["POLCO"] * 3 + ["ZEAMX"] * 2 + ["other_plant"] * 2
    predicted = (["CHEAL", "CHEAL", "CHEAL", "MATCH"] + ["POLCO", "STEME", "STEME"]
                 + ["ZEAMX", "ZEAMX"] + ["CHEAL", "CHEAL"])
    datasets = (["acre", "acre", "oppd", "oppd"] + ["cropandweed"] * 3 + ["cropandweed"] * 2
                + ["oppd", "acre"])
    train = Counter({("CHEAL", "acre"): 80, ("CHEAL", "oppd"): 20, ("POLCO", "oppd"): 90,
                     ("POLCO", "cropandweed"): 10, ("ZEAMX", "cropandweed"): 50,
                     ("other_plant", "oppd"): 5, ("other_plant", "acre"): 5})
    rows, summary = class_by_dataset(true, predicted, datasets, train, "other_plant", 0.5)
    cells = {(r["category"], r["dataset"]): r for r in rows}
    assert ("ZEAMX", "cropandweed") not in cells  # one dataset only
    assert cells["CHEAL", "oppd"] == {
        "category": "CHEAL", "dataset": "oppd", "crops": 2, "accuracy": 0.5, "train_crops": 20,
        "train_share": 0.2, "most_wrong": "MATCH 50%"}
    assert cells["POLCO", "cropandweed"]["most_wrong"] == "STEME 67%"
    # cross-dataset: CHEAL in oppd (2 crops, 1 right) and POLCO in cropandweed (3, 1 right);
    # other_plant is left out
    assert summary == {"cross_dataset_crops": 5, "cross_dataset_accuracy": pytest.approx(0.4)}


def test_warning_when_the_last_epoch_is_clearly_below_the_best():
    records = [{"epoch": 0, "val_accuracy": 0.90}, {"epoch": 1, "val_accuracy": 0.937},
               {"epoch": 2, "val_accuracy": 0.936}]
    assert last_below_best(records, 0.005) is None  # 0.1 point: chance
    records.append({"epoch": 3, "val_accuracy": 0.925})
    warning = last_below_best(records, 0.005)
    assert warning.startswith("Warning: the last epoch (3)") and "best (epoch 1" in warning
    assert last_below_best([], 0.005) is None


def test_partial_label_loss():
    torch.manual_seed(0)
    logits, labels = torch.randn(6, 4), torch.tensor([0, 3, 1, 1, 2, 0])
    one_hot = torch.nn.functional.one_hot(labels, 4).bool()
    for smoothing in (0.0, 0.1):  # one allowed class: the usual cross-entropy
        expected = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)(logits, labels)
        assert torch.allclose(PartialLabelLoss(smoothing)(logits, one_hot), expected)
    allowed = torch.tensor([[True, False, True, False]])
    probs = logits[:1].softmax(1)[0]
    assert torch.allclose(PartialLabelLoss()(logits[:1], allowed),
                          -torch.log(probs[0] + probs[2]))


def test_sample_weights_balance_classes():
    labels = [0, 0, 0, 0, 1]
    assert sample_weights(labels, "none") is None
    inverse = sample_weights(labels, "inverse")
    assert sum(inverse[:4]) == pytest.approx(inverse[4])  # both classes equally likely
    sqrt = sample_weights(labels, "sqrt")
    assert sum(sqrt[:4]) == pytest.approx(2 * sqrt[4])  # in proportion to sqrt(4) : sqrt(1)
    with pytest.raises(ValueError):
        sample_weights(labels, "equal")


def test_dinov3_builds_with_the_same_code():
    # patch 16 (DINOv2: 14) and rotary position embeddings, through the same timm interface
    model = build_classifier(3, {**TINY_MODEL, "backbone": "vit_small_patch16_dinov3"}, 32)
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 3)
    frozen = build_classifier(3, {**TINY_MODEL, "backbone": "vit_small_patch16_dinov3",
                                  "freeze_blocks": 12}, 32)
    assert {name for name, p in frozen.named_parameters() if p.requires_grad} == {
        "norm.weight", "norm.bias", "head.weight", "head.bias"}


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
    partly = np.concatenate([probs, [[0.5, 0.5]]]), np.concatenate([labels, [-1]])
    assert accuracy_metrics(*partly, 2) == metrics  # partial labels are left out
    assert label_fit(np.array([0, 1, 1]), [[0, 2], [0, 2], [1]]) == pytest.approx(2 / 3)
    assert label_fit(np.array([]), []) is None


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


def test_train_one_epoch_with_partial_labels():
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, -1] * 4)
    allowed = torch.nn.functional.one_hot(labels.clamp(min=0), 3).bool()
    allowed[labels < 0] = torch.tensor([True, True, False])  # "not class 2"
    data = torch.utils.data.TensorDataset(torch.randn(16, 5), labels, allowed)
    model = torch.nn.Linear(5, 3)
    with torch.no_grad():  # always predicts class 0
        model.weight.zero_()
        model.bias.copy_(torch.tensor([5.0, 0.0, 0.0]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    loader = torch.utils.data.DataLoader(data, 4)
    result = train_one_epoch(model, loader, optimizer, scheduler,
                             torch.amp.GradScaler("cpu", enabled=False), PartialLabelLoss(),
                             torch.device("cpu"), False, None, 0)
    assert result["accuracy"] == pytest.approx(1 / 3)  # of the 12 labelled samples
    assert result["loss"] > 0
