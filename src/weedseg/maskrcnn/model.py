"""torchvision Mask R-CNN (ResNet-50 FPN v2) set up for our data."""

from typing import Any

import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import batched_nms, box_iou, clip_boxes_to_image, remove_small_boxes

from weedseg.paths import resolve


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


class ClassScoreRoIHeads(RoIHeads):
    """RoI heads that keep the full class distribution of each detection.

    torchvision turns every proposal into one detection per class and keeps only that
    class's score. This is its postprocess_detections with the proposal of each detection
    tracked, so `class_scores` holds, per image of the last batch, a tensor [detections,
    classes incl. background] of softmax probabilities, in the order of the detections.
    """

    class_scores: list[torch.Tensor]

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(p) for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals).split(boxes_per_image, 0)
        pred_scores = F.softmax(class_logits, -1).split(boxes_per_image, 0)
        all_boxes, all_scores, all_labels = [], [], []
        self.class_scores = []
        for boxes, probs, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=probs.device).expand_as(probs)
            rows = torch.arange(len(probs), device=probs.device)[:, None].expand_as(probs)
            # one candidate per proposal and class, without background
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores, labels, rows = (t[:, 1:].reshape(-1) for t in (probs, labels, rows))
            keep = torch.where(scores > self.score_thresh)[0]
            keep = keep[remove_small_boxes(boxes[keep], min_size=1e-2)]
            keep = keep[batched_nms(boxes[keep], scores[keep], labels[keep], self.nms_thresh)]
            keep = keep[: self.detections_per_img]
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])
            all_labels.append(labels[keep])
            self.class_scores.append(probs[rows[keep]])
        return all_boxes, all_scores, all_labels


def build_model(num_classes: int, config: dict[str, Any]) -> MaskRCNN:
    """Mask R-CNN with `num_classes` including background (= categories + 1).

    With `pretrained`, all weights except the class-specific heads start from torchvision's
    COCO model: read from `pretrained_path` when set, else downloaded once to the torch cache.
    """
    if len(config["anchor_sizes"]) != 5:
        raise ValueError("anchor_sizes needs one size per FPN level (5)")
    weights, weights_path = None, None
    if config["pretrained"]:
        if config["pretrained_path"]:
            weights_path = resolve(config["pretrained_path"])
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {weights_path}. Download "
                    f"{MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.url} there, or set "
                    "model.pretrained_path to null to download them to the torch cache")
        else:
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=tuple(config["min_size"]),
        max_size=config["max_size"],
        box_detections_per_img=config["detections_per_image"],
        box_score_thresh=config["score_threshold"],
    )
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    # One size per FPN level and three aspect ratios, as in the default generator, so the
    # pretrained RPN head still fits.
    model.rpn.anchor_generator = AnchorGenerator(
        tuple((size,) for size in config["anchor_sizes"]),
        ((0.5, 1.0, 2.0),) * len(config["anchor_sizes"]))
    model.rpn.__class__ = IgnoreAwareRPN
    model.roi_heads.__class__ = ClassScoreRoIHeads

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
