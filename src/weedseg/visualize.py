"""Drawing COCO annotations (and later predictions) over images, for visual checks."""

from typing import Any

import cv2
import numpy as np

from weedseg.data.coco import IGNORE_CATEGORY, decode_mask

IGNORE_COLOR = (160, 160, 160)  # BGR


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


def draw_annotations(image: np.ndarray, annotations: list[dict[str, Any]],
                     categories: list[dict[str, Any]], mask_alpha: float = 0.45,
                     draw_boxes: bool = True, draw_labels: bool = True,
                     draw_stems: bool = True) -> np.ndarray:
    """Return a copy of a BGR image with masks, boxes, labels and stem points drawn on.

    Works for ground truth and for predictions (COCO results, whose `score` joins the label).
    Ignore regions (iscrowd) are grey and hatched, with a dashed-looking thin box, so they
    are easy to tell apart from real instances.
    """
    height, width = image.shape[:2]
    names = {c["id"]: c["name"] for c in categories}
    colors = category_colors(categories)
    scale = max(height, width) / 1920  # line and text sizes relative to a 1920 px image
    thickness = max(1, round(2 * scale))
    font_scale = 0.6 * scale

    overlay = image.copy()
    hatch = np.zeros((height, width), bool)
    diagonal = np.add.outer(np.arange(height), np.arange(width))
    hatch_pattern = diagonal % max(8, round(16 * scale)) < max(2, round(3 * scale))
    contours = []
    for annotation in annotations:
        color = colors.get(annotation["category_id"], IGNORE_COLOR)
        crowd = bool(annotation.get("iscrowd"))
        mask = decode_mask(annotation, height, width)
        if mask is not None:
            if crowd:
                overlay[mask] = IGNORE_COLOR
                hatch |= mask & hatch_pattern
            else:
                overlay[mask] = color
            found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            contours.append((found, IGNORE_COLOR if crowd else color))
    result = cv2.addWeighted(overlay, mask_alpha, image, 1 - mask_alpha, 0)
    result[hatch] = (60, 60, 60)
    for found, color in contours:
        cv2.drawContours(result, found, -1, color, thickness)

    for annotation in annotations:
        crowd = bool(annotation.get("iscrowd"))
        color = IGNORE_COLOR if crowd else colors.get(annotation["category_id"], IGNORE_COLOR)
        x, y, w, h = (round(v) for v in annotation["bbox"])
        if draw_boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 1 if crowd else thickness)
        if draw_stems and annotation.get("stem") and not crowd:
            sx, sy = (round(v) for v in annotation["stem"])
            cv2.drawMarker(result, (sx, sy), (255, 255, 255), cv2.MARKER_CROSS,
                           max(8, round(14 * scale)), thickness + 1)
            cv2.drawMarker(result, (sx, sy), color, cv2.MARKER_CROSS,
                           max(8, round(14 * scale)), thickness)
        if draw_labels and not crowd:
            text = names.get(annotation["category_id"], "?")
            if "score" in annotation:  # a prediction
                text += f" {annotation['score']:.2f}"
            put_label(result, text, (x, y), color, font_scale, thickness)
    return result


def plot_confusion_matrix(matrix: np.ndarray, names: list[str], path, title: str) -> None:
    """Save a confusion matrix as PNG: colour = share of the row (ground-truth class),
    text = count. Rows are ground truth, columns predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # One-hue sequential ramp, from the surface colour to dark blue.
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    size = max(4.0, 0.42 * len(names) + 2)
    fig, ax = plt.subplots(figsize=(size + 1.2, size), facecolor="#fcfcfb")
    image = ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90 if len(names) > 6 else 0,
                  color="#52514e", fontsize=9)
    ax.set_yticks(range(len(names)), names, color="#52514e", fontsize=9)
    ax.set_xlabel("Predicted", color="#0b0b0b")
    ax.set_ylabel("Ground truth", color="#0b0b0b")
    ax.set_title(title, color="#0b0b0b", loc="left", fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(names) <= 30:
        for row in range(len(names)):
            for col in range(len(names)):
                if matrix[row, col]:
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                            fontsize=8 if len(names) > 10 else 10,
                            color="#f0efec" if shares[row, col] > 0.55 else "#0b0b0b")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Share of ground-truth row", color="#52514e")
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def put_label(image: np.ndarray, text: str, corner: tuple[int, int],
              color: tuple[int, int, int], font_scale: float, thickness: int) -> None:
    """Text on a filled background whose bottom-left is at `corner` (moved inside the image)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = min(max(corner[0], 0), image.shape[1] - tw - 4)
    y = max(corner[1], th + baseline + 4)
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(image, text, (x + 2, y - baseline - 2), font, font_scale, text_color,
                thickness, cv2.LINE_AA)
