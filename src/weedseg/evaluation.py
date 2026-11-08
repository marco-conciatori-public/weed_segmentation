"""COCO box evaluation of detections against our internal COCO format.

Model independent: detections are COCO results (`image_id`, `category_id`, `bbox` as
x, y, w, h, and `score`).

Our ignore regions (iscrowd) belong to the `ignore` category, or keep the category of their
label when it has one. Plain COCOeval matches crowd regions per category, so it would count a
detection of another category on an ignore region as a false positive. `IgnoreAwareCOCOeval`
copies every ignore region into every evaluated category instead.
"""

import contextlib
import io
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from weedseg.data.coco import IGNORE_CATEGORY

# AP and AR at max_detections unless named otherwise (AR1, AR10); s/m/l are COCO's area
# ranges (under 32x32 px, up to 96x96 px, above).
SUMMARY_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl",
                 "AR1", "AR10", "ARmax", "ARs", "ARm", "ARl"]


def ignore_category_ids(gt: dict[str, Any]) -> set[int]:
    return {c["id"] for c in gt["categories"] if c["name"] == IGNORE_CATEGORY}


class IgnoreAwareCOCOeval(COCOeval):
    """COCOeval (boxes) whose ignore regions act as crowd regions for every category."""

    def __init__(self, coco_gt: COCO, coco_dt: COCO, max_detections: int = 100):
        super().__init__(coco_gt, coco_dt, "bbox")
        self.ignore_ids = ignore_category_ids(coco_gt.dataset)
        self.params.catIds = [c for c in sorted(coco_gt.getCatIds()) if c not in self.ignore_ids]
        self.params.maxDets = [1, 10, max_detections]

    def _prepare(self) -> None:
        super()._prepare()
        annotations = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.params.imgIds))
        regions = [a for a in annotations
                   if a.get("iscrowd") or a["category_id"] in self.ignore_ids]
        for region in regions:
            for category_id in self.params.catIds:
                if category_id == region["category_id"]:  # COCOeval already has it there
                    continue
                self._gts[region["image_id"], category_id].append(
                    {**region, "category_id": category_id, "iscrowd": 1, "ignore": 1})


def coco_from_dict(data: dict[str, Any]) -> COCO:
    coco = COCO()
    coco.dataset = data
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def evaluate_coco(gt: dict[str, Any], results: list[dict[str, Any]],
                  image_ids: list[int] | None = None, max_detections: int = 100,
                  verbose: bool = False) -> dict[str, Any]:
    """COCO box metrics: {"summary": {...}, "per_class": [...]}.

    `image_ids` restricts the evaluation to those images (default: all images in `gt`).
    `per_class` lists every evaluated category with its number of (non-ignore) ground-truth
    instances and its AP and AP50 (NaN when the category has no instances).
    """
    coco_gt = coco_from_dict(gt)
    image_ids = sorted(image_ids if image_ids is not None else coco_gt.getImgIds())
    selected = set(image_ids)
    gt_counts = defaultdict(int)
    for annotation in gt["annotations"]:
        if annotation["image_id"] in selected and not annotation.get("iscrowd"):
            gt_counts[annotation["category_id"]] += 1
    names = {c["id"]: c["name"] for c in gt["categories"]}

    dts = [r for r in results if r["image_id"] in selected]
    if not dts:
        return {"summary": dict.fromkeys(SUMMARY_NAMES, 0.0), "per_class": []}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dts)
        evaluator = IgnoreAwareCOCOeval(coco_gt, coco_dt, max_detections)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
    summary = _summary(evaluator)
    if verbose:
        print("box: " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()))

    # precision: [IoU thresholds, recall points, categories, area ranges, max dets]
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    per_class = []
    for k, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, k]
        per_class.append({
            "category": names[category_id],
            "gt_instances": gt_counts[category_id],
            "AP": _mean_valid(values),
            "AP50": _mean_valid(values[0]),
        })
    return {"summary": summary, "per_class": per_class}


def _summary(evaluator: COCOeval) -> dict[str, float]:
    """COCOeval.summarize() without printing, and at our max_detections: summarize() takes
    AP at 100 detections, so with any other limit its main AP comes out as -1."""
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    iou_index = {round(float(t), 2): i for i, t in enumerate(evaluator.params.iouThrs)}
    values = [
        precision[:, :, :, 0, -1], precision[iou_index[0.5], :, :, 0, -1],
        precision[iou_index[0.75], :, :, 0, -1], precision[:, :, :, 1, -1],
        precision[:, :, :, 2, -1], precision[:, :, :, 3, -1],
        recall[:, :, 0, 0], recall[:, :, 0, 1], recall[:, :, 0, -1],
        recall[:, :, 1, -1], recall[:, :, 2, -1], recall[:, :, 3, -1],
    ]
    return {name: _mean_valid(v, empty=0.0) for name, v in zip(SUMMARY_NAMES, values)}


def _mean_valid(values: np.ndarray, empty: float = float("nan")) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else empty


@dataclass
class Match:
    """A labelled plant and the detection matched to it; either may be None (a missed plant,
    or a detection on no labelled plant)."""

    image_id: int
    plant: dict[str, Any] | None
    detection: dict[str, Any] | None


def match_detections(gt: dict[str, Any], results: list[dict[str, Any]],
                     image_ids: list[int] | None = None, score_threshold: float = 0.5,
                     iou_threshold: float = 0.5) -> list[Match]:
    """Pair detections with score >= `score_threshold` with labelled plants.

    Matching ignores classes: greedily by score, by box IoU. Unmatched detections that lie on
    an ignore region (IoA >= `iou_threshold`) are dropped.
    """
    ignore_ids = ignore_category_ids(gt)
    image_ids = set(image_ids if image_ids is not None else (i["id"] for i in gt["images"]))
    gts, regions, dts = defaultdict(list), defaultdict(list), defaultdict(list)
    for annotation in gt["annotations"]:
        if annotation["image_id"] in image_ids:
            crowd = annotation.get("iscrowd") or annotation["category_id"] in ignore_ids
            (regions if crowd else gts)[annotation["image_id"]].append(annotation)
    for result in results:
        if result["image_id"] in image_ids and result["score"] >= score_threshold:
            dts[result["image_id"]].append(result)

    matches = []
    for image_id in sorted(image_ids):
        image_gts, image_dts = gts[image_id], sorted(dts[image_id], key=lambda r: -r["score"])
        matched, matched_dts = set(), set()
        if image_gts and image_dts:
            ious = _ious(image_dts, image_gts, crowd=False)
            for d, dt in enumerate(image_dts):
                candidates = [(ious[d, g], g) for g in range(len(image_gts))
                              if g not in matched and ious[d, g] >= iou_threshold]
                if candidates:
                    g = max(candidates)[1]
                    matched.add(g)
                    matched_dts.add(d)
                    matches.append(Match(image_id, image_gts[g], dt))
        unmatched = [dt for d, dt in enumerate(image_dts) if d not in matched_dts]
        if unmatched and regions[image_id]:
            ioas = _ious(unmatched, regions[image_id], crowd=True)
            unmatched = [dt for d, dt in enumerate(unmatched) if ioas[d].max() < iou_threshold]
        matches += [Match(image_id, None, dt) for dt in unmatched]
        matches += [Match(image_id, annotation, None)
                    for g, annotation in enumerate(image_gts) if g not in matched]
    return matches


def _ious(dts: list[dict[str, Any]], gts: list[dict[str, Any]], crowd: bool) -> np.ndarray:
    """Box IoU matrix [detections, ground truth]; with `crowd`, IoA of the detections instead."""
    d = [r["bbox"] for r in dts]
    g = [a["bbox"] for a in gts]
    return np.asarray(mask_utils.iou(d, g, [int(crowd)] * len(g))).reshape(len(dts), len(gts))
