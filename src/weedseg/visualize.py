"""Drawing COCO annotations and model predictions over images, for visual checks."""

from typing import Any

import cv2
import numpy as np

from weedseg.data.coco import IGNORE_CATEGORY

IGNORE_COLOR = (160, 160, 160)  # BGR
PLANT_COLOR = (80, 220, 80)  # detections of an exported detector


def category_colors(categories: list[dict[str, Any]]) -> dict[int, tuple[int, int, int]]:
    """A distinct, stable BGR colour per category id (golden-ratio hues); ignore is grey."""
    colors = {}
    for i, category in enumerate(categories):
        if category["name"] == IGNORE_CATEGORY:
            colors[category["id"]] = IGNORE_COLOR
            continue
        hue = int((i * 0.618034 % 1.0) * 180)  # OpenCV hue range is 0-179
        hsv = np.uint8([[[hue, 220, 255]]])
        colors[category["id"]] = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    return colors


def _sizes(image: np.ndarray) -> tuple[float, int, float]:
    """Scale, line thickness and font scale for an image, relative to a 1920 px one."""
    scale = max(image.shape[:2]) / 1920
    thickness = max(1, round(2 * scale))
    return scale, thickness, 0.6 * scale


def draw_annotations(image: np.ndarray, annotations: list[dict[str, Any]],
                     categories: list[dict[str, Any]], draw_labels: bool = True,
                     draw_stems: bool = True) -> np.ndarray:
    """Return a copy of a BGR image with boxes, labels and stem points drawn on.

    Works for ground truth and for predictions (COCO results, whose `score` joins the label).
    A plant's label is its own EPPO code (`eppo`, kept when datasets are combined into one
    plant class), else its category. Ignore regions (iscrowd) are grey, hatched boxes, so
    they are easy to tell apart from plants.
    """
    names = {c["id"]: c["name"] for c in categories}
    colors = category_colors(categories)
    scale, thickness, font_scale = _sizes(image)
    result = image.copy()
    height, width = image.shape[:2]
    period, line = max(8, round(16 * scale)), max(2, round(3 * scale))
    for annotation in annotations:
        if not annotation.get("iscrowd"):
            continue
        x, y, w, h = (round(v) for v in annotation["bbox"])
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        diagonal = np.add.outer(np.arange(y0, y1), np.arange(x0, x1))
        result[y0:y1, x0:x1][diagonal % period < line] = (60, 60, 60)
        cv2.rectangle(result, (x, y), (x + w, y + h), IGNORE_COLOR, 1)

    for annotation in annotations:
        if annotation.get("iscrowd"):
            continue
        color = colors.get(annotation["category_id"], IGNORE_COLOR)
        x, y, w, h = (round(v) for v in annotation["bbox"])
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        if draw_stems and annotation.get("stem"):
            sx, sy = (round(v) for v in annotation["stem"])
            cv2.drawMarker(result, (sx, sy), (255, 255, 255), cv2.MARKER_CROSS,
                           max(8, round(14 * scale)), thickness + 1)
            cv2.drawMarker(result, (sx, sy), color, cv2.MARKER_CROSS,
                           max(8, round(14 * scale)), thickness)
        if draw_labels:
            text = annotation.get("eppo") or names.get(annotation["category_id"], "?")
            if "score" in annotation:  # a prediction
                text += f" {annotation['score']:.2f}"
            put_label(result, text, (x, y), color, font_scale, thickness)
    return result


def draw_detections(image: np.ndarray, plants: list[dict[str, Any]],
                    labels: bool = True) -> np.ndarray:
    """Return a copy of a BGR image with the plants an exported detector found drawn on
    (weedseg.deploy.plant_detector: `box` and `plant_score`). Labels are placed so they don't
    cover each other, the most certain plants' first."""
    _, thickness, font_scale = _sizes(image)
    result = image.copy()
    for plant in plants:
        x, y, w, h = (round(v) for v in plant["box"])
        cv2.rectangle(result, (x, y), (x + w, y + h), PLANT_COLOR, thickness)
    if labels:
        taken = []
        for plant in plants:
            x, y = (round(v) for v in plant["box"][:2])
            put_label(result, f"{plant['plant_score']:.2f}", (x, y), PLANT_COLOR, font_scale,
                      thickness, taken)
    return result


def put_label(image: np.ndarray, text: str, corner: tuple[int, int],
              color: tuple[int, int, int], font_scale: float, thickness: int,
              taken: list[tuple[int, int, int, int]] | None = None) -> None:
    """Text on a filled background whose bottom-left is at `corner` (moved inside the image).

    `taken`: the rectangles (x0, y0, x1, y1) of labels already drawn. The label moves down
    below any it would cover (while it fits in the image), and its own is added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = min(max(corner[0], 0), image.shape[1] - tw - 4)
    y = max(corner[1], th + baseline + 4)
    if taken is not None:
        label_height = th + baseline + 4
        while True:
            covered = [r for r in taken
                       if x < r[2] and r[0] < x + tw + 4 and y - label_height < r[3] and r[1] < y]
            if not covered or max(r[3] for r in covered) + label_height > image.shape[0]:
                break
            y = max(r[3] for r in covered) + label_height
        taken.append((x, y - label_height, x + tw + 4, y))
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(image, text, (x + 2, y - baseline - 2), font, font_scale, text_color,
                thickness, cv2.LINE_AA)
