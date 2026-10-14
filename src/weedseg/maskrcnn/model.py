"""torchvision Mask R-CNN (ResNet-50 FPN v2) set up for our data."""

from typing import Any

import torch
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import box_iou


class IgnoreAwareRPN(RegionProposalNetwork):
    """RPN that neither rewards nor penalises anchors matched to ignore regions.

    Targets label ignore regions -1. The RoI heads already skip proposals matched to a -1
    box (their sampler only takes labels >= 1 and == 0); the stock RPN looks at boxes only
    and would train such anchors as objects, so here they get the "between thresholds"
    label -1 as well.
    """

    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched_gt_boxes = super().assign_targets_to_anchors(anchors, targets)
        for anchors_in_image, target, labels_in_image in zip(anchors, targets, labels):
            ignore = target["labels"] < 0
            if target["boxes"].numel() == 0 or not ignore.any():
                continue
            matched = self.proposal_matcher(box_iou(target["boxes"], anchors_in_image))
            on_ignore = (matched >= 0) & ignore[matched.clamp(min=0)]
            labels_in_image[on_ignore] = -1.0
        return labels, matched_gt_boxes


def build_model(num_classes: int, config: dict[str, Any]) -> MaskRCNN:
    """Mask R-CNN with `num_classes` including background (= categories + 1).

    With `pretrained`, all weights except the class-specific heads start from torchvision's
    COCO model (downloaded once to the torch cache).
    """
    if len(config["anchor_sizes"]) != 5:
        raise ValueError("anchor_sizes needs one size per FPN level (5)")
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if config["pretrained"] else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=tuple(config["min_size"]),
        max_size=config["max_size"],
        box_detections_per_img=config["detections_per_image"],
        box_score_thresh=config["score_threshold"],
    )
    # One size per FPN level and three aspect ratios, as in the default generator, so the
    # pretrained RPN head still fits.
    model.rpn.anchor_generator = AnchorGenerator(
        tuple((size,) for size in config["anchor_sizes"]),
        ((0.5, 1.0, 2.0),) * len(config["anchor_sizes"]))
    model.rpn.__class__ = IgnoreAwareRPN

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def load_checkpoint_model(path, device: torch.device) -> tuple[MaskRCNN, dict[str, Any]]:
    """Model from a checkpoint saved by scripts/train_maskrcnn.py, plus the checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = {**checkpoint["config"]["model"], "pretrained": False}
    model = build_model(len(checkpoint["categories"]) + 1, config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint
