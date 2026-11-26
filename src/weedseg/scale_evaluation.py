"""How many plants does the plant detector still find when a photo shows the ground at another
scale than its training images did? This tells whether an estimate of the ground resolution
(mm/px) is worth building for photos whose scale nobody gives: phones, drones, other robots.

The detector resizes every image so its short side is `min_size` px (1088; the long side at most
`max_size`, 1920) before it looks. So the scale it sees depends on how much ground the photo
covers, not on the camera: mm/px at the detector = the image's mm/px / that resize. Its training
images ran from 0.41 (CropAndWeed) to 0.94 mm/px (PhenoBench). Each val image is shown at each of
`targets` (mm/px at the detector) instead:

- coarser than its usual scale: the whole image, shrunk by area averaging (what a camera with
  fewer pixels on the ground records). The detector's input is then smaller. A real photo would
  hold more ground around the same plants, which a detector working locally barely notices (one
  difference: its detections per image would be shared by more plants).
- finer: a centred window, enlarged to the detector's usual input size: a close-up. Plants cut by
  the window's edge are left out (ignore regions). A window that would enlarge the image's own
  pixels more than `max_upsample` times is skipped: a real close-up would be sharp, an enlarged
  one is blurred.

Two more ways to show an image:

- `rescaled`: shrunk to the target, then enlarged back to its usual scale. This is a coarse photo
  whose ground resolution is known (estimated) and corrected by resizing. A photo with pixels to
  spare would instead be cut into tiles at the usual scale, which the `native` row shows.
- `deployed`: at its usual scale, but resized by the detector's own transform (bilinear, no
  anti-aliasing), as the exported model does, instead of by area averaging. Large photos (OPPD,
  GroundBased, any phone) are shrunk 3 to 4 times by it.

A plant counts as found when a detection with plant score >= `score_threshold` overlaps its box
at IoU >= `iou_threshold`. Boxes, not masks, so the datasets labelled with boxes only (new to the
detector) count too. Box AP (COCO) is free of the threshold. Detections are kept in a cache, so
an interrupted run resumes. scripts/experiments/evaluate_scale.py runs it. See docs/twostage.md (Ground
resolution and the detector).
"""

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from weedseg.config import load_config
from weedseg.data.coco import IGNORE_CATEGORY
from weedseg.evaluation import evaluate_coco, match_detections
from weedseg.maskrcnn.data import load_coco, load_masked_coco
from weedseg.maskrcnn.engine import select_device
from weedseg.maskrcnn.model import load_checkpoint_model
from weedseg.maskrcnn.run import checkpoint_path
from weedseg.paths import resolve
from weedseg.skew_evaluation import write_csv

PLANT, IGNORE = 1, 2  # category ids of the plant truth
CATEGORIES = [{"id": PLANT, "name": "plant"}, {"id": IGNORE, "name": IGNORE_CATEGORY}]
NATIVE = None  # the target of an image's usual scale
VARIANTS = {"as_is": "as is", "rescaled": "rescaled", "deployed": "deployed resize"}
IMAGE_KEYS = ("id", "file_name", "width", "height", "mm_per_px", "dataset")
# The detector on an image (RGB, uint8) -> detections [n, 5]: x, y, w, h, score in its px.
# With True, the image is resized by the detector's own transform, else taken as it is.
Detect = Callable[[np.ndarray, bool], np.ndarray]


@dataclass(frozen=True)
class View:
    """How the detector is shown an image: its `window` (x0, y0, x1, y1 in the image's px; None =
    all of it) resized by `factor`, after a shrink by `degrade` when set (rescaled); `deployed`:
    resized to `factor` by the detector's own transform."""

    factor: float
    window: tuple[int, int, int, int] | None = None
    degrade: float | None = None
    deployed: bool = False

    @property
    def key(self) -> str:
        """Its cache key: two cases with the same view share their detections."""
        return json.dumps([round(self.factor, 6), list(self.window) if self.window else None,
                           None if self.degrade is None else round(self.degrade, 6),
                           self.deployed])

    @property
    def detail(self) -> float:
        """Pixels of the image's detail per pixel of the image (the shrink, when degraded)."""
        return self.factor if self.degrade is None else min(self.factor, self.degrade)


def usual_factor(width: int, height: int, min_size: int, max_size: int) -> float:
    """The detector's own resize of an image: short side to `min_size`, long side at most
    `max_size`."""
    return min(min_size / min(width, height), max_size / max(width, height))


def plan(image: dict[str, Any], variant: str, target: float | None, sizes: tuple[int, int],
         max_upsample: float) -> View | None:
    """The view of `image` (width, height, mm_per_px) that puts it at `target` mm/px at the
    detector (NATIVE: its usual scale), or None when `variant` doesn't show it there."""
    width, height = image["width"], image["height"]
    usual = usual_factor(width, height, *sizes)
    if target is NATIVE:
        if variant == "rescaled":
            return None
        if variant == "deployed" and usual != 1:  # at 1 both leave the image as it is
            return View(usual, deployed=True)
        return View(usual)
    if variant == "deployed":
        return None
    factor = image["mm_per_px"] / target
    if variant == "rescaled":
        return View(usual, degrade=factor) if factor < usual else None
    if factor <= usual:
        return View(factor)
    if factor > max_upsample:
        return None
    zoom = factor / usual
    w, h = round(width / zoom), round(height / zoom)
    x0, y0 = (width - w) // 2, (height - h) // 2
    return View(factor, (x0, y0, x0 + w, y0 + h))


def cases(settings: dict[str, Any]) -> list[tuple[str, float | None]]:
    """(variant, target) of every row, in report order."""
    found = [("as_is", NATIVE), *(("as_is", t) for t in settings["targets"])]
    if settings["deployed"]:
        found.append(("deployed", NATIVE))
    return found + [("rescaled", t) for t in settings["rescaled"]]


def output_size(width: int, height: int, factor: float) -> tuple[int, int]:
    return max(1, round(width * factor)), max(1, round(height * factor))


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`image` at `size` (width, height): area averaging to shrink, bilinear to enlarge."""
    height, width = image.shape[:2]
    if (width, height) == size:
        return image
    shrink = size[0] * size[1] < width * height
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def show(image: np.ndarray, view: View) -> np.ndarray:
    """What the detector is given for `view` of an image."""
    if view.window is not None:
        x0, y0, x1, y1 = view.window
        image = image[y0:y1, x0:x1]
    if view.deployed:
        return image
    height, width = image.shape[:2]
    if view.degrade is not None:
        image = resize(image, output_size(width, height, view.degrade))
    return resize(image, output_size(width, height, view.factor))


def to_image(detections: np.ndarray, view: View, shown: tuple[int, int],
             window: tuple[int, int]) -> np.ndarray:
    """Detections [n, 5] on the shown image (`shown`: its width, height) in the image's px
    (`window`: the width and height of the part shown)."""
    found = detections.astype(np.float32, copy=True)
    found[:, [0, 2]] *= window[0] / shown[0]
    found[:, [1, 3]] *= window[1] / shown[1]
    if view.window is not None:
        found[:, 0] += view.window[0]
        found[:, 1] += view.window[1]
    return found


def detector_function(model: torch.nn.Module, device: torch.device, amp: bool,
                      sizes: tuple[int, int]) -> Detect:
    """`Detect` with a Mask R-CNN: boxes only (its mask head is dropped, which leaves the boxes
    as they are and saves time and memory)."""
    model.roi_heads.mask_roi_pool = model.roi_heads.mask_head = None
    model.roi_heads.mask_predictor = None
    model.eval()

    @torch.inference_mode()
    def detect(image: np.ndarray, deployed: bool) -> np.ndarray:
        height, width = image.shape[:2]
        if deployed:
            model.transform.min_size, model.transform.max_size = (sizes[0],), sizes[1]
        else:  # a resize by 1: the image as it is
            model.transform.min_size = (min(height, width),)
            model.transform.max_size = max(height, width)
        # mul(1 / 255): bit for bit the scaling of training (as weedseg.deploy does)
        tensor = (torch.from_numpy(np.ascontiguousarray(image)).to(device).permute(2, 0, 1)
                  .float().mul(1 / 255))
        with torch.autocast(device.type, enabled=amp):
            output = model([tensor])[0]
        boxes = output["boxes"].float().cpu().numpy()
        scores = output["scores"].float().cpu().numpy()
        return np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2], scores])
    return detect


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_cache(path: Path, images: dict[int, dict[str, Any]]) -> dict[tuple[int, str], np.ndarray]:
    """Detections kept by an earlier run, by (image id, view key); lines of another image under
    the same id (another split file) are left out, and so is a line cut short by a killed run."""
    found = {}
    if not path.exists():
        return found
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            image = images.get(entry["image"])
            if image is not None and image["file_name"] == entry["file"]:
                found[entry["image"], entry["view"]] = np.asarray(
                    entry["detections"], np.float32).reshape(-1, 5)
    return found


def free_memory_mb() -> float | None:
    """The machine's available memory in MB, or None without psutil (then nothing is checked)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 2**20


def detect_all(images: list[dict[str, Any]], plans: dict[Any, dict[int, View | None]],
               image_dir: Path, detect: Detect, cache: Path, min_free_mb: float | None = None,
               free_mb: Callable[[], float | None] = free_memory_mb,
               ) -> dict[tuple[int, str], np.ndarray]:
    """Detections of every view of every image, by (image id, view key): from `cache` when it
    has them, else predicted and added to it. Each image is read once for all its views.

    Before each image, stops (SystemExit) when the machine has less than `min_free_mb` of
    memory available, to leave it to other work; the detections so far stay in `cache`, so
    running again resumes."""
    found = load_cache(cache, {image["id"]: image for image in images})
    todo = []
    for image in images:
        views = {view.key: view for views in plans.values()
                 if (view := views.get(image["id"])) is not None}
        views = [view for key, view in views.items() if (image["id"], key) not in found]
        if views:
            todo.append((image, views))
    if not todo:
        return found
    print(f"detector: {sum(len(v) for _, v in todo)} views of {len(todo)} images to predict "
          f"({len(found)} kept in {cache})")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cut = False
    if cache.exists() and cache.stat().st_size:
        with open(cache, "rb") as f:
            f.seek(-1, 2)
            cut = f.read(1) != b"\n"
    with open(cache, "a", encoding="utf-8") as f:
        if cut:  # a killed run's last line: start a new one, so it alone is lost
            f.write("\n")
        for done, (image, views) in enumerate(tqdm(todo, desc="detect", dynamic_ncols=True)):
            free = free_mb() if min_free_mb is not None else None
            if free is not None and free < min_free_mb:
                raise SystemExit(f"Stopped: {free:.0f} MB of memory free, under min_free_memory_mb "
                                 f"({min_free_mb:g}), after {done} of {len(todo)} images. The "
                                 f"detections so far are in {cache}: run again to resume.")
            rgb = read_rgb(image_dir / image["file_name"])
            for view in views:
                shown = show(rgb, view)
                window = ((view.window[2] - view.window[0], view.window[3] - view.window[1])
                          if view.window else (rgb.shape[1], rgb.shape[0]))
                detections = to_image(detect(shown, view.deployed), view,
                                      (shown.shape[1], shown.shape[0]), window)
                found[image["id"], view.key] = detections
                rows = [[round(v, 1) for v in d[:4]] + [round(d[4], 4)]
                        for d in detections.tolist()]
                f.write(json.dumps({"image": image["id"], "file": image["file_name"],
                                    "view": view.key, "detections": rows}) + "\n")
                f.flush()
    return found


def choose_images(coco: dict[str, Any], datasets: list[str], plants_per_dataset: int | None,
                  max_images: int | None, seed: int) -> list[dict[str, Any]]:
    """Per dataset, images in a fixed random order until they hold `plants_per_dataset`
    labelled plants or are `max_images` (None = no limit)."""
    ignore_ids = {c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY}
    plants = Counter(a["image_id"] for a in coco["annotations"]
                     if not a.get("iscrowd") and a["category_id"] not in ignore_ids)
    by_dataset = defaultdict(list)
    for image in coco["images"]:
        by_dataset[image.get("dataset")].append({k: image.get(k) for k in IMAGE_KEYS})
    missing = [name for name in datasets if name not in by_dataset]
    if missing:
        raise SystemExit(f"Datasets not in the split file: {missing}")
    chosen = []
    for name in datasets:
        pool = sorted(by_dataset[name], key=lambda image: image["id"])
        random.Random(seed).shuffle(pool)
        count = 0
        for taken, image in enumerate(pool):
            if ((plants_per_dataset is not None and count >= plants_per_dataset)
                    or (max_images is not None and taken >= max_images)):
                break
            if not image["mm_per_px"]:
                raise SystemExit(f"{image['file_name']} has no mm_per_px")
            chosen.append(image)
            count += plants[image["id"]]
    return chosen


def plant_truth(coco: dict[str, Any], images: list[dict[str, Any]]) -> dict[int, list[dict]]:
    """Per image id: its labelled boxes, each a plant or an ignore region (crowd, or the ignore
    category): every species is a plant."""
    ignore_ids = {c["id"] for c in coco["categories"] if c["name"] == IGNORE_CATEGORY}
    truth = {image["id"]: [] for image in images}
    for a in coco["annotations"]:
        if a["image_id"] in truth:
            ignore = bool(a.get("iscrowd")) or a["category_id"] in ignore_ids
            x, y, w, h = a["bbox"]
            truth[a["image_id"]].append({"id": a["id"], "image_id": a["image_id"],
                                         "bbox": [x, y, w, h], "area": w * h,
                                         "iscrowd": int(ignore),
                                         "category_id": IGNORE if ignore else PLANT})
    return truth


def in_window(annotations: list[dict[str, Any]], window: tuple[int, int, int, int]) -> list[dict]:
    """The boxes a window shows: those whole inside as they are, those it cuts as ignore regions
    (clipped to it); those outside are left out."""
    x0, y0, x1, y1 = window
    kept = []
    for a in annotations:
        bx, by, bw, bh = a["bbox"]
        cx0, cy0 = max(bx, x0), max(by, y0)
        cx1, cy1 = min(bx + bw, x1), min(by + bh, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if bx >= x0 and by >= y0 and bx + bw <= x1 and by + bh <= y1:
            kept.append(a)
        else:
            w, h = cx1 - cx0, cy1 - cy0
            kept.append({**a, "bbox": [cx0, cy0, w, h], "area": w * h, "iscrowd": 1,
                         "category_id": IGNORE})
    return kept


def band_names(edges: list[float]) -> list[str]:
    """Names of the plant size bands cut at `edges` (mm)."""
    return ([f"under {edges[0]:g} mm"]
            + [f"{a:g}-{b:g} mm" for a, b in pairwise(edges)]
            + [f"{edges[-1]:g} mm and over"])


def score(images: list[dict[str, Any]], truth: dict[int, list[dict]],
          views: dict[int, View | None], detections: dict[int, np.ndarray],
          settings: dict[str, Any], max_detections: int) -> dict[str, Any]:
    """The numbers of one set of images, each shown by its view (None: not shown);
    `detections` by image id, in the image's px."""
    shown = [image for image in images if views.get(image["id"]) is not None]
    annotations, results = [], []
    for image in shown:
        view = views[image["id"]]
        boxes = truth[image["id"]]
        annotations += in_window(boxes, view.window) if view.window else boxes
        results += [{"image_id": image["id"], "category_id": PLANT, "bbox": d[:4], "score": d[4]}
                    for d in detections[image["id"]].tolist()]
    values = {"images": len(shown)}
    if not shown:
        return values
    coco = {"images": [{"id": i["id"], "width": i["width"], "height": i["height"]}
                       for i in shown], "annotations": annotations, "categories": CATEGORIES}
    matches = match_detections(coco, results, score_threshold=settings["score_threshold"],
                               iou_threshold=settings["iou_threshold"], match_by="box")
    plants = [m for m in matches if m.plant is not None]
    false = sum(m.plant is None for m in matches)
    found = np.array([m.detection is not None for m in plants], bool)
    longest = np.array([max(m.plant["bbox"][2:]) for m in plants])
    by_id = {image["id"]: image for image in shown}
    size_mm = longest * np.array([by_id[m.image_id]["mm_per_px"] for m in plants])
    detail_px = longest * np.array([views[m.image_id].detail for m in plants])
    values |= {"plants": len(plants),
               "plant_px": float(np.median(detail_px)) if len(plants) else None,
               "found": float(found.mean()) if len(plants) else None}
    edges = settings["size_bands_mm"]
    bands = np.digitize(size_mm, edges)
    for k, name in enumerate(band_names(edges)):
        here = bands == k
        values[f"found {name}"] = float(found[here].mean()) if here.any() else None
        values[f"plants {name}"] = int(here.sum())
    hits = int(found.sum())
    values |= {"false_per_image": false / len(shown),
               "precision": hits / (hits + false) if hits + false else None}
    summary = evaluate_coco(coco, results, iou_types=("bbox",),
                            max_detections=max_detections)["bbox"]["summary"]
    return values | {"AP50": summary["AP50"], "AP": summary["AP"]}


def score_all(images: list[dict[str, Any]], truth: dict[int, list[dict]],
              plans: dict[Any, dict[int, View | None]],
              detections: dict[tuple[int, str], np.ndarray], settings: dict[str, Any],
              sizes: tuple[int, int], max_detections: int) -> list[dict[str, Any]]:
    """One row per dataset and case (variant, target)."""
    rows = []
    for dataset in settings["datasets"]:
        here = [image for image in images if image["dataset"] == dataset]
        usual = [usual_factor(i["width"], i["height"], *sizes) for i in here]
        native = float(np.median([i["mm_per_px"] / f for i, f in zip(here, usual)]))
        for (variant, target), views in plans.items():
            arrays = {i["id"]: detections[i["id"], views[i["id"]].key]
                      for i in here if views.get(i["id"]) is not None}
            mm_px = native if target is NATIVE else target
            rows.append({"dataset": dataset, "trained_on": dataset in settings["trained_on"],
                         "variant": variant, "target": "native" if target is NATIVE else target,
                         "detector_mm_px": mm_px, "ground_m": mm_px * sizes[0] / 1000,
                         "usual_factor": float(np.median(usual)),
                         **score(here, truth, views, arrays, settings, max_detections)})
    return rows


def saved_check(detector_file: Path, images: list[dict[str, Any]], truth: dict[int, list[dict]],
                plans: dict[Any, dict[int, View | None]],
                detections: dict[tuple[int, str], np.ndarray], settings: dict[str, Any],
                max_detections: int) -> list[str]:
    """The native rows against the detector's saved evaluation (GPU, mixed precision, the
    deployed resize), on the images both have, matched by file name."""
    folder = detector_file.parent / "eval" / f"val_{detector_file.stem}"
    if not (folder / "predictions.json").exists():
        return [f"no saved evaluation in {folder} to check against"]
    split = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))["split_file"]
    # the evaluation leaves out the images labelled with boxes only, as load_masked_coco does
    names = {i["id"]: i["file_name"] for i in load_masked_coco(resolve(split))["images"]}
    ours = {image["file_name"]: image["id"] for image in images}
    covered = {ours[name] for name in names.values() if name in ours}
    saved = defaultdict(list)
    with open(folder / "predictions.json", encoding="utf-8") as f:
        for p in json.load(f):
            image_id = ours.get(names.get(p["image_id"]))
            if image_id is not None:
                saved[image_id].append([*p["bbox"], p["score"]])

    def cached(views: dict[int, View | None], here: list[dict]) -> dict[int, np.ndarray]:
        return {i["id"]: detections[i["id"], views[i["id"]].key] for i in here}

    native = [(VARIANTS[v], plans[v, NATIVE]) for v in ("deployed", "as_is")
              if (v, NATIVE) in plans]
    lines = []
    for dataset in settings["datasets"]:
        here = [i for i in images if i["dataset"] == dataset and i["id"] in covered]
        if not here:
            continue
        arrays = {i["id"]: np.asarray(saved[i["id"]], np.float32).reshape(-1, 5) for i in here}
        first = score(here, truth, native[0][1], arrays, settings, max_detections)
        parts = [f"{percent(first['found'])}% in the saved evaluation"]
        for name, views in native:
            values = score(here, truth, views, cached(views, here), settings, max_detections)
            parts.append(f"{percent(values['found'])}% here ({name})")
        lines.append(f"{dataset} ({first['plants']} plants of {len(here)} images): found "
                     + ", ".join(parts))
    return lines or ["the saved evaluation has none of these images"]


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def _row(rows: list[dict[str, Any]], dataset: str, variant: str, target: Any) -> dict | None:
    return next((r for r in rows if r["dataset"] == dataset and r["variant"] == variant
                 and r["target"] == target), None)


def _found(row: dict[str, Any] | None) -> str:
    return "-" if row is None or row.get("found") is None else percent(row["found"])


def report(rows: list[dict[str, Any]], settings: dict[str, Any], checks: list[str],
           detector: str, sizes: tuple[int, int]) -> str:
    """The Markdown report: plants found per dataset and scale, the rescaled and deployed
    comparisons, then every number per dataset."""
    datasets = settings["datasets"]
    new = [d for d in datasets if d not in settings["trained_on"]]
    names = [f"{d} (new)" if d in new else d for d in datasets]
    intro = (
        f"Detector: {detector}. Plants found: a detection with plant score >= "
        f"{settings['score_threshold']} on the plant's box at IoU >= {settings['iou_threshold']}, "
        "in % of the labelled plants shown. (new) = no image of that dataset was in the "
        "detector's training (boxes only, or kept out). The scale is mm per pixel at the "
        f"detector, which shrinks or enlarges every image to {sizes[0]} px on its short side "
        f"(and at most {sizes[1]} on its long side), so 1 mm/px is about "
        f"{sizes[0] / 1000:.2f} m of ground across the short side. Coarser targets shrink the "
        "image by area averaging; finer ones show a centred window enlarged to the usual input "
        "size (a close-up), skipped (-) where it would enlarge the image's own pixels more than "
        f"{settings['max_upsample']:g} times. See docs/twostage.md (Ground resolution and the "
        "detector).")
    lines = [
        intro, "",
        "Checks of the native rows against the detector's saved evaluation:", "",
        *(f"- {check}" for check in checks), "",
        "# Plants found, as is", "",
        "| mm/px at the detector | ground across the short side | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(datasets)]
    native = [_row(rows, d, "as_is", "native") for d in datasets]
    lines.append("| native | | " + " | ".join(
        f"{_found(r)} ({r['detector_mm_px']:.2f})" for r in native) + " |")
    for target in settings["targets"]:
        lines.append(f"| {target:g} | {target * sizes[0] / 1000:.2f} m | " + " | ".join(
            _found(_row(rows, d, "as_is", target)) for d in datasets) + " |")
    lines += ["", "Native: each image at its own scale, the dataset's median in brackets.", ""]

    if settings["rescaled"]:
        note = ("As is -> shrunk to the target and enlarged back to the image's usual scale "
                "(what a known ground resolution allows for a photo with no pixels to spare). "
                "A photo with pixels to spare can be tiled at the usual scale: the native row.")
        lines += ["# Coarse photos rescaled to the usual scale", "", note, "",
                  "| mm/px | " + " | ".join(names) + " |", "|---|" + "---|" * len(datasets)]
        for target in settings["rescaled"]:
            lines.append(f"| {target:g} | " + " | ".join(
                f"{_found(_row(rows, d, 'as_is', target))} -> "
                f"{_found(_row(rows, d, 'rescaled', target))}" for d in datasets) + " |")
        lines.append("")

    if settings["deployed"]:
        note = ("Native scale, area averaging -> the detector's own resize (bilinear, no "
                "anti-aliasing), as the exported model resizes a photo.")
        lines += ["# The deployed resize", "", note, "",
                  "| dataset | resize | found | box AP50 | false detections per image |",
                  "|---|---|---|---|---|"]
        for dataset, name in zip(datasets, names):
            area, deployed = (_row(rows, dataset, v, "native") for v in ("as_is", "deployed"))
            lines.append(
                f"| {name} | x{area['usual_factor']:.2f} | {_found(area)} -> {_found(deployed)} "
                f"| {_ap(area)} -> {_ap(deployed)} | {_false(area)} -> {_false(deployed)} |")
        lines.append("")

    bands = band_names(settings["size_bands_mm"])
    columns = {"images": "images", "plants": "plants", "plant_px": "median plant px",
               "found": "found", **{f"found {b}": b for b in bands},
               "false_per_image": "false per image", "precision": "precision",
               "AP50": "box AP50", "AP": "box AP"}
    note = ("Median plant px: the plant's longest side in pixels of detail (after the shrink, "
            "for rescaled). Found by the plant's real size, longest side. False: detections on "
            "no labelled plant and no ignore region.")
    lines += ["# Per dataset", "", note, ""]
    for dataset, name in zip(datasets, names):
        lines += [f"## {name}", "", "| case | mm/px | " + " | ".join(columns.values()) + " |",
                  "|---|---|" + "---|" * len(columns)]
        for row in (r for r in rows if r["dataset"] == dataset):
            cells = []
            for key in columns:
                value = row.get(key)
                if value is None:
                    cells.append("-")
                elif key in ("images", "plants"):
                    cells.append(str(value))
                elif key == "plant_px":
                    cells.append(f"{value:.0f}")
                elif key == "false_per_image":
                    cells.append(f"{value:.2f}")
                elif key.startswith("AP"):
                    cells.append(f"{value:.3f}")
                else:
                    cells.append(percent(value))
            lines.append(f"| {VARIANTS[row['variant']]} | {row['detector_mm_px']:.2f} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _ap(row: dict[str, Any] | None) -> str:
    return "-" if row is None or row.get("AP50") is None else f"{row['AP50']:.3f}"


def _false(row: dict[str, Any] | None) -> str:
    return "-" if row is None or row.get("false_per_image") is None else (
        f"{row['false_per_image']:.2f}")


def write_report(settings: dict[str, Any], stem: Path) -> str:
    """Predict every view, score it, write <stem>.md and <stem>.csv; returns the Markdown."""
    torch.set_num_threads(settings["threads"])
    cv2.setNumThreads(settings["threads"])
    twostage = load_config(settings["twostage"])
    detector_file = checkpoint_path(load_config(twostage["detector"]["config"]),
                                    twostage["detector"]["checkpoint"])
    split = resolve(settings["split_file"])
    coco = load_coco(split)
    images = choose_images(coco, settings["datasets"], settings["plants_per_dataset"],
                           settings["max_images_per_dataset"], settings["seed"])
    truth = plant_truth(coco, images)
    image_dir = resolve(coco["info"]["image_dir"])
    del coco

    device = select_device(settings["device"])
    model, checkpoint = load_checkpoint_model(detector_file, device)
    model_config = checkpoint["config"]["model"]
    sizes = (model_config["min_size"][-1], model_config["max_size"])
    plans = {(variant, target): {image["id"]: plan(image, variant, target, sizes,
                                                   settings["max_upsample"]) for image in images}
             for variant, target in cases(settings)}
    print(f"{len(images)} images: " + ", ".join(
        f"{n} {d}" for d, n in Counter(i["dataset"] for i in images).items()))
    cache = (resolve(settings["output_dir"]) / f"{detector_file.parent.name}_{detector_file.stem}"
             / f"{split.parent.parent.name}_{split.parent.name}_{split.stem}.jsonl")
    detect = detector_function(model, device, settings["amp"] and device.type == "cuda", sizes)
    detections = detect_all(images, plans, image_dir, detect, cache,
                            settings["min_free_memory_mb"])
    del model, detect

    max_detections = model_config["detections_per_image"]
    rows = score_all(images, truth, plans, detections, settings, sizes, max_detections)
    checks = saved_check(detector_file, images, truth, plans, detections, settings,
                         max_detections)
    text = report(rows, settings, checks, f"{detector_file.parent.name}/{detector_file.name}",
                  sizes)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    return text
