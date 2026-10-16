"""COCO evaluation of instance predictions against our internal COCO format.

Model independent: predictions are COCO results (`image_id`, `category_id`, `bbox` as
x, y, w, h, `score` and optionally an RLE `segmentation`).

Our ignore regions (iscrowd) belong to the `ignore` category, or keep the category of their
label when it has one. Plain COCOeval matches crowd regions per category, so it would count a
detection of another category on an ignore region as a false positive. `IgnoreAwareCOCOeval`
copies every ignore region into every evaluated category instead.
"""

import contextlib
import io
from collections import defaultdict
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
    """COCOeval whose ignore regions act as crowd regions for every category."""

    def __init__(self, coco_gt: COCO, coco_dt: COCO, iou_type: str, max_detections: int = 100):
        super().__init__(coco_gt, coco_dt, iou_type)
        self.ignore_ids = ignore_category_ids(coco_gt.dataset)
        self.params.catIds = [c for c in sorted(coco_gt.getCatIds()) if c not in self.ignore_ids]
        self.params.maxDets = [1, 10, max_detections]

    def _prepare(self) -> None:
        super()._prepare()
        annotations = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.params.imgIds))
        regions = [a for a in annotations
                   if a.get("iscrowd") or a["category_id"] in self.ignore_ids]
        for region in regions:
            if self.params.iouType == "segm":
                region = {**region, "segmentation": self.cocoGt.annToRLE(region)}
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
                  image_ids: list[int] | None = None, iou_types: tuple[str, ...] = ("bbox", "segm"),
                  max_detections: int = 100, verbose: bool = False) -> dict[str, Any]:
    """COCO metrics per IoU type: {"bbox": {"summary": {...}, "per_class": [...]}, ...}.

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

    metrics = {}
    for iou_type in iou_types:
        dts = [r for r in results if r["image_id"] in selected]
        if iou_type == "segm":
            # Without a bbox, loadRes takes areas from the masks (for the APs/APm/APl ranges).
            dts = [{k: v for k, v in r.items() if k != "bbox"} for r in dts if "segmentation" in r]
        if not dts:
            metrics[iou_type] = {"summary": dict.fromkeys(SUMMARY_NAMES, 0.0), "per_class": []}
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(dts)
            evaluator = IgnoreAwareCOCOeval(coco_gt, coco_dt, iou_type, max_detections)
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
        summary = _summary(evaluator)
        if verbose:
            print(f"{iou_type}: " + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()))

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
        metrics[iou_type] = {"summary": summary, "per_class": per_class}
    return metrics


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


def confusion_matrix(gt: dict[str, Any], results: list[dict[str, Any]],
                     image_ids: list[int] | None = None, score_threshold: float = 0.5,
                     iou_threshold: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Class confusion of detections with score >= `score_threshold`.

    Detections are matched to ground truth regardless of class (greedily by score, IoU of
    masks when present, else boxes). Rows are ground-truth classes, columns predicted
    classes; the last row/column is "background": a missed plant (last column) or a
    detection on no plant (last row). Unmatched detections that lie on an ignore region
    (IoA >= `iou_threshold`) are not counted.
    """
    ignore_ids = ignore_category_ids(gt)
    categories = [c for c in gt["categories"] if c["id"] not in ignore_ids]
    index = {c["id"]: i for i, c in enumerate(categories)}
    background = len(categories)
    matrix = np.zeros((background + 1, background + 1), np.int64)

    image_ids = set(image_ids if image_ids is not None else (i["id"] for i in gt["images"]))
    sizes = {i["id"]: (i["height"], i["width"]) for i in gt["images"]}
    gts, regions, dts = defaultdict(list), defaultdict(list), defaultdict(list)
    for annotation in gt["annotations"]:
        if annotation["image_id"] in image_ids:
            crowd = annotation.get("iscrowd") or annotation["category_id"] in ignore_ids
            (regions if crowd else gts)[annotation["image_id"]].append(annotation)
    for result in results:
        if result["image_id"] in image_ids and result["score"] >= score_threshold:
            dts[result["image_id"]].append(result)

    for image_id in image_ids:
        image_gts, image_dts = gts[image_id], sorted(dts[image_id], key=lambda r: -r["score"])
        use_masks = bool(image_dts) and all("segmentation" in r for r in image_dts)
        height, width = sizes[image_id]
        matched, matched_dts = set(), set()
        if image_gts and image_dts:
            ious = _ious(image_dts, image_gts, use_masks, height, width, crowd=False)
            for d, dt in enumerate(image_dts):
                candidates = [(ious[d, g], g) for g in range(len(image_gts))
                              if g not in matched and ious[d, g] >= iou_threshold]
                if candidates:
                    g = max(candidates)[1]
                    matched.add(g)
                    matched_dts.add(d)
                    matrix[index[image_gts[g]["category_id"]], index[dt["category_id"]]] += 1
        unmatched = [dt for d, dt in enumerate(image_dts) if d not in matched_dts]
        if unmatched and regions[image_id]:
            ioas = _ious(unmatched, regions[image_id], use_masks, height, width, crowd=True)
            unmatched = [dt for d, dt in enumerate(unmatched) if ioas[d].max() < iou_threshold]
        for dt in unmatched:
            matrix[background, index[dt["category_id"]]] += 1
        for g, annotation in enumerate(image_gts):
            if g not in matched:
                matrix[index[annotation["category_id"]], background] += 1
    return matrix, [c["name"] for c in categories] + ["background"]


def _ious(dts: list[dict[str, Any]], gts: list[dict[str, Any]], use_masks: bool,
          height: int, width: int, crowd: bool) -> np.ndarray:
    """IoU matrix [detections, ground truth]; with `crowd`, IoA of the detections instead."""
    if use_masks:
        d = [r["segmentation"] for r in dts]
        g = [_rle(a, height, width) for a in gts]
    else:
        d = [r["bbox"] for r in dts]
        g = [a["bbox"] for a in gts]
    return np.asarray(mask_utils.iou(d, g, [int(crowd)] * len(g))).reshape(len(dts), len(gts))


def _rle(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation.get("segmentation")
    if not segmentation:  # box only: use the box as mask
        x, y, w, h = annotation["bbox"]
        return mask_utils.frPyObjects([[x, y, x + w, y, x + w, y + h, x, y + h]], height, width)[0]
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation["counts"], list):
        return mask_utils.frPyObjects(segmentation, height, width)
    return segmentation
