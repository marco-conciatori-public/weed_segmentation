"""Does a plant's place against the crop rows (weedseg.rows) help tell crop from weed, once the
classifier has had its say? Measured on saved predictions: no GPU and no retraining.

Three sets of plants:

- position alone: every labelled plant, with rows from the labelled crops. No model: how much
  the position says at all, per dataset and crop.
- two-stage: the detections of a two-stage evaluation, with the classifier's probabilities.
- unseen: labelled boxes of datasets a run never trained on, with the probabilities that
  scripts/evaluate_unseen.py saves. The test of a new site. Only crops and weeds with a label are
  in it, so it is as if the detector were perfect.

How much a position is worth is learned, as a table of log likelihood ratios per cell (how far
off the row, how close to the next plant, how far from an expected plant), from plants other
than those judged: see FOLDS. It joins the classifier in two ways, and neither touches the
order of the classes inside the crops or inside the rest:

- plain: odds of a crop x ratio ^ trust, as if the classifier's probabilities could be taken at
  their word. Label smoothing keeps them far from 0 and 1, so they can't.
- stack: a logistic regression on the classifier's logit and the ratio, learned on the same
  plants as the table. The fairest version: it can give the position no weight at all.

A position counts only in an image whose fullest row has enough other plants, since any two
plants make a line. See docs/twostage.md (Crop rows and spacing).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from weedseg.compare import run_dir, twostage_eval
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import ACTIONS, Rules, harmful_removed_at, load_rules, role_of
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.data import load_coco
from weedseg.paths import resolve
from weedseg.rows import POSITIONS, Geometry, RowSettings, plant_positions, session_geometry
from weedseg.skew_evaluation import load_evaluation, spread, write_csv

# What a labelled plant is. An edge crop is a crop cut by the image border, an ignore region: it
# shows where a row runs, but no score counts it.
CROP, EDGE_CROP, WEED, OTHER = "crop", "edge_crop", "weed", "other"
# Which plants the rows come from.
EVIDENCE = ("called_crop", "labelled", "all")
# Whom a position's worth is learned from, for the plants of a session:
# - other_sessions: every other session, of any dataset;
# - same_dataset: the other sessions of its dataset: tuned to the crop and the place;
# - other_datasets: other datasets only: all a new field could have.
FOLDS = ("other_sessions", "same_dataset", "other_datasets")
VARIANTS = ("rows", "rows+spacing", "size")
CLASSIFIER = "classifier"
STACK_C = 1000.0  # regularisation of the stack: next to none


@dataclass
class Plants:
    """The plants of one evaluation; per plant unless said otherwise. Without a classifier
    (position alone), `probs` is None and there are no classes."""

    probs: np.ndarray | None  # [plants, classes], each row summing to 1
    is_crop: np.ndarray  # per class
    roles: list[str]  # per class
    rules: Rules | None
    plant_scores: np.ndarray
    points: np.ndarray  # [plants, 2], in mm
    sizes: np.ndarray  # the longer side of the box, in mm
    datasets: np.ndarray
    sessions: np.ndarray  # dataset/group
    frames: np.ndarray  # image id
    labels: np.ndarray  # CROP, EDGE_CROP, WEED, OTHER, or "" on no labelled plant
    harmful: np.ndarray  # labelled with a harmful species

    @property
    def p_crop(self) -> np.ndarray:
        return self.probs[:, self.is_crop].sum(1)

    @property
    def scored(self) -> np.ndarray:
        """Plants that a table may learn from and a score counts."""
        return np.isin(self.labels, (CROP, WEED))


def label_of(annotation: dict[str, Any], tag: str | None) -> str:
    """What a labelled plant is, by its annotation and its category's supercategory. Weeds the
    source didn't name are ignore regions marked hidden_plant, known as weeds by their source
    label. Hidden plants named only by genus or family aren't counted: they could be anything."""
    if tag == "crop":
        return EDGE_CROP if annotation.get("iscrowd") else CROP
    if annotation.get("iscrowd"):
        hidden_weed = (annotation.get("hidden_plant") and annotation.get("allowed_classes")
                       and "weed" in str(annotation.get("source_label", "")).lower())
        return WEED if hidden_weed else ""
    return {"weed": WEED, "other": OTHER}.get(tag, "")


def _place(image: dict[str, Any], box: list[float]) -> tuple[tuple[float, float], float]:
    """A plant's position and size in mm: the centre and longer side of its box."""
    mm = image["mm_per_px"]
    return ((box[0] + box[2] / 2) * mm, (box[1] + box[3] / 2) * mm), max(box[2], box[3]) * mm


def _session(image: dict[str, Any]) -> str:
    return f"{image.get('dataset')}/{image.get('group')}"


def labelled_plants(gt: dict[str, Any]) -> Plants:
    """Every labelled plant of a ground truth, without a classifier."""
    images = {image["id"]: image for image in gt["images"]}
    tags = {c["id"]: c.get("supercategory") for c in gt["categories"]}
    labelled = [(a, label_of(a, tags[a["category_id"]])) for a in gt["annotations"]]
    labelled = [(a, label) for a, label in labelled if label]
    places = [_place(images[a["image_id"]], a["bbox"]) for a, _ in labelled]
    return Plants(
        None, np.empty(0, bool), [], None, np.ones(len(labelled)),
        np.array([point for point, _ in places], dtype=np.float64).reshape(-1, 2),
        np.array([size for _, size in places], dtype=np.float64),
        np.array([images[a["image_id"]].get("dataset") for a, _ in labelled], dtype=str),
        np.array([_session(images[a["image_id"]]) for a, _ in labelled], dtype=str),
        np.array([a["image_id"] for a, _ in labelled], dtype=np.int64),
        np.array([label for _, label in labelled], dtype=str), np.zeros(len(labelled), bool))


def _box_iou(box: list[float], boxes: np.ndarray) -> np.ndarray:
    width = (np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
             - np.maximum(box[0], boxes[:, 0])).clip(0)
    height = (np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
              - np.maximum(box[1], boxes[:, 1])).clip(0)
    overlap = width * height
    return overlap / np.maximum(box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - overlap, 1e-9)


def twostage_plants(folder: Path, ground_truths: dict[Path, dict], iou: float = 0.5) -> Plants:
    """The detections of a two-stage evaluation. Named plants are matched as the evaluation does;
    a detection left over takes its label from a hidden weed or an edge crop it overlaps by box
    (`iou`), which the evaluation leaves out, as they are ignore regions."""
    found = load_evaluation(folder, ground_truths)
    gt = found.gt
    images = {image["id"]: image for image in gt["images"]}
    tags = {c["id"]: c.get("supercategory") for c in gt["categories"]}
    hidden = defaultdict(list)
    for annotation in gt["annotations"]:
        label = label_of(annotation, tags[annotation["category_id"]])
        if annotation.get("iscrowd") and label:
            hidden[annotation["image_id"]].append((annotation["bbox"], label))
    labels, harmful = [], []
    for row, image_id in enumerate(found.image_ids):
        true = int(found.true[row])
        if true >= 0:
            name = found.classes[true]
            labels.append(CROP if found.is_crop[true]
                          else OTHER if name == OTHER_CATEGORY else WEED)
            harmful.append(found.roles[name] == "harmful")
            continue
        label = ""
        if hidden.get(image_id):
            overlaps = _box_iou(found.boxes[row], np.array([b for b, _ in hidden[image_id]], float))
            if overlaps.max() >= iou:
                label = hidden[image_id][int(overlaps.argmax())][1]
        labels.append(label)
        harmful.append(False)
    places = [_place(images[image_id], box) for image_id, box in zip(found.image_ids, found.boxes)]
    return Plants(
        found.probs, found.is_crop, [found.roles[name] for name in found.classes], found.rules,
        found.plant_scores,
        np.array([point for point, _ in places], dtype=np.float64).reshape(-1, 2),
        np.array([size for _, size in places], dtype=np.float64),
        np.array([dataset for dataset, _ in found.sessions], dtype=str),
        np.array([f"{dataset}/{group}" for dataset, group in found.sessions], dtype=str),
        np.array(found.image_ids, dtype=np.int64), np.array(labels, dtype=str),
        np.array(harmful, dtype=bool))


def unseen_plants(path: Path, gt: dict[str, Any], datasets: list[str], rules: Rules,
                  taxa: dict[str, dict[str, str]]) -> Plants:
    """Labelled boxes of `datasets` with a run's probabilities, as evaluate_unseen.py saves them
    (annotation_ids, probs, classes). The run's classes may differ from the data's: what is a crop
    goes by the data's categories, so a species it has no class for is simply not a crop to it."""
    saved = np.load(path)
    classes = saved["classes"].tolist()
    images = {image["id"]: image for image in gt["images"]}
    annotations = {a["id"]: a for a in gt["annotations"]}
    tag_of = {c["id"]: c.get("supercategory") for c in gt["categories"]}
    crops = {c["name"] for c in gt["categories"] if c.get("supercategory") == "crop"}
    chosen = [k for k, annotation_id in enumerate(saved["annotation_ids"].tolist())
              if images[annotations[annotation_id]["image_id"]].get("dataset") in datasets]
    plants = [annotations[int(saved["annotation_ids"][k])] for k in chosen]
    categories = {c["id"]: c["name"] for c in gt["categories"]}
    places = [_place(images[a["image_id"]], a["bbox"]) for a in plants]
    roles = [role_of(name, rules, taxa) for name in classes]
    return Plants(
        saved["probs"][chosen], np.array([name in crops for name in classes]), roles, rules,
        np.ones(len(plants)),
        np.array([point for point, _ in places], dtype=np.float64).reshape(-1, 2),
        np.array([size for _, size in places], dtype=np.float64),
        np.array([images[a["image_id"]].get("dataset") for a in plants], dtype=str),
        np.array([_session(images[a["image_id"]]) for a in plants], dtype=str),
        np.array([a["image_id"] for a in plants], dtype=np.int64),
        np.array([label_of(a, tag_of[a["category_id"]]) for a in plants], dtype=str),
        np.array([categories[a["category_id"]] in taxa
                  and role_of(categories[a["category_id"]], rules, taxa) == "harmful"
                  for a in plants], dtype=bool))


def evidence_of(plants: Plants, source: str) -> np.ndarray:
    """Which plants the rows come from: those the classifier calls a crop, those labelled as
    crops (the best a classifier could do), or all (rows that don't trust the classifier)."""
    if source == "called_crop":
        return plants.p_crop >= 0.5
    if source == "labelled":
        return np.isin(plants.labels, (CROP, EDGE_CROP))
    if source == "all":
        return np.ones(len(plants.labels), bool)
    raise ValueError(f"The rows come from one of {EVIDENCE}, not {source!r}")


def judge_positions(plants: Plants, source: str,
                    settings: RowSettings) -> tuple[np.ndarray, dict[str, Geometry]]:
    """Every plant's POSITIONS, and the geometry of each session that has one."""
    evidence = evidence_of(plants, source)
    found = np.full(len(plants.labels), np.nan, dtype=POSITIONS)
    sessions = defaultdict(lambda: defaultdict(list))
    for row, (session, frame) in enumerate(zip(plants.sessions, plants.frames.tolist())):
        sessions[str(session)][frame].append(row)
    geometries = {}
    for session, frames in sessions.items():
        geometry = session_geometry([plants.points[[row for row in rows if evidence[row]]]
                                     for rows in frames.values()], settings)
        if geometry is None:
            continue
        geometries[session] = geometry
        for rows in frames.values():
            found[rows] = plant_positions(plants.points[rows], evidence[rows], geometry,
                                          plants.sizes[rows], settings)
    return found, geometries


def _bucket(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """The interval of each value among `edges`; len(edges) + 1 when it isn't known."""
    return np.where(np.isnan(values), len(edges) + 1, np.digitize(values, edges))


def table_cells(positions: np.ndarray, variant: str, cells: dict[str, Any],
                min_row_plants: int) -> np.ndarray:
    """The cell of each plant for a table, as a number; -1 when its image has no real row, so
    that the position says nothing. Spacing counts only for a plant on a row."""
    distance = _bucket(positions["row_distance"], cells["row_distance_mm"])
    on_row = positions["row_distance"] < cells["spacing_within_mm"]
    spacing = np.where(on_row, _bucket(positions["spacing"], cells["spacing"]), 9)
    phase = np.where(on_row, _bucket(positions["phase"], cells["phase"]), 9)
    if variant == CLASSIFIER:  # no position at all: one cell, whose ratio is 0
        found = np.zeros(len(positions), dtype=np.int64)
    elif variant == "rows":
        found = distance
    elif variant == "rows+spacing":
        found = distance + 100 * spacing + 10000 * phase
    elif variant == "size":
        found = _bucket(positions["size_ratio"], cells["size"])
    else:
        raise ValueError(f"The tables are {VARIANTS}, not {variant!r}")
    return np.where(positions["strongest_row"] >= min_row_plants, found, -1)


def log_ratio_table(cells: np.ndarray, is_crop: np.ndarray, smooth: float,
                    largest_ratio: float) -> dict[int, float]:
    """log P(cell | crop) - log P(cell | not crop), per cell seen, smoothed by adding `smooth`
    plants to each cell of both and limited to `largest_ratio`, either way."""
    values, inverse = np.unique(cells, return_inverse=True)
    crops = np.bincount(inverse[is_crop], minlength=len(values)) + smooth
    others = np.bincount(inverse[~is_crop], minlength=len(values)) + smooth
    ratios = np.log(crops / crops.sum()) - np.log(others / others.sum())
    limit = np.log(largest_ratio)
    return dict(zip(values.tolist(), ratios.clip(-limit, limit).tolist()))


def look_up(table: dict[int, float], cells: np.ndarray) -> np.ndarray:
    """The table's value per cell; 0, saying nothing, for a cell it hasn't seen."""
    values, inverse = np.unique(cells, return_inverse=True)
    return np.array([table.get(value, 0.0) for value in values.tolist()])[inverse]


def logit(p: np.ndarray) -> np.ndarray:
    p = p.clip(1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x.clip(-30, 30)))


def fold_masks(plants: Plants, fold: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(plants judged, plants learned from), for every part of FOLDS' `fold`. The two never
    share a session, and for other_datasets never a dataset."""
    if fold == "other_datasets":
        return [(plants.datasets == dataset, plants.datasets != dataset)
                for dataset in np.unique(plants.datasets)]
    if fold not in FOLDS:
        raise ValueError(f"Learning is from one of {FOLDS}, not {fold!r}")
    masks = []
    for session in np.unique(plants.sessions):
        judged = plants.sessions == session
        learned = ~judged
        if fold == "same_dataset":
            learned = learned & (plants.datasets == plants.datasets[judged][0])
        masks.append((judged, learned))
    return masks


@dataclass
class Learned:
    """What a position's worth is learned from: labelled plants in an image with a real row."""

    cells: np.ndarray
    base: np.ndarray  # logit of the classifier's P(crop)
    is_crop: np.ndarray


def learned_from(plants: Plants, cells: np.ndarray, base: np.ndarray,
                 mask: np.ndarray | None = None) -> Learned:
    keep = plants.scored & (cells != -1)
    if mask is not None:
        keep = keep & mask
    return Learned(cells[keep], base[keep], plants.labels[keep] == CROP)


def combine(base: np.ndarray, cells: np.ndarray, parts: list[tuple[np.ndarray, Learned]],
            settings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """For every plant, its log ratio and the stack's logit, each part judged only by what it
    learned from. Also the stack's mean weights (classifier, position). A part with too little
    to learn from keeps what the classifier says."""
    ratios, stack, weights = np.zeros(len(base)), base.copy(), []
    for judged, learned in parts:
        if min(learned.is_crop.sum(), (~learned.is_crop).sum()) < settings["min_to_learn"]:
            continue
        table = log_ratio_table(learned.cells, learned.is_crop, settings["smooth"],
                                settings["largest_ratio"])
        model = LogisticRegression(C=STACK_C, class_weight="balanced", max_iter=1000).fit(
            np.column_stack([learned.base, look_up(table, learned.cells)]), learned.is_crop)
        ratios[judged] = look_up(table, cells[judged])
        stack[judged] = model.decision_function(np.column_stack([base[judged], ratios[judged]]))
        weights.append(model.coef_[0])
    return ratios, stack, np.mean(weights, 0) if weights else None


def rescaled(plants: Plants, p_crop: np.ndarray) -> np.ndarray:
    """The probabilities with the crops summing to `p_crop`, each side keeping its proportions."""
    old = plants.p_crop
    rest = 1 - old
    probs = plants.probs.copy()
    probs[:, plants.is_crop] *= np.divide(p_crop, old, out=np.ones_like(old),
                                          where=old > 0)[:, None]
    probs[:, ~plants.is_crop] *= np.divide(1 - p_crop, rest, out=np.ones_like(old),
                                           where=rest > 0)[:, None]
    total = probs.sum(1, keepdims=True)
    return np.divide(probs, total, out=probs, where=total > 0)


def removable(plants: Plants, probs: np.ndarray) -> np.ndarray:
    """As weedseg.decisions.removed_up_to, for all plants: the highest defer_below that removes
    each, NaN when none does."""
    actions = np.array([ACTIONS[role] for role in plants.roles])
    remove = probs[:, actions == "remove"].sum(1)
    keep = probs[:, actions == "keep"].sum(1)
    gate = plants.rules.plant_defer_below or 0
    return np.where((remove > keep) & (plants.plant_scores >= gate), remove, np.nan)


def _known(values: np.ndarray) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values]


def _auc(crops: np.ndarray, others: np.ndarray) -> float | None:
    """How often a crop scores above another plant; NaN scores are left out."""
    crops, others = crops[~np.isnan(crops)], others[~np.isnan(others)]
    if not len(crops) or not len(others):
        return None
    return float(roc_auc_score(np.r_[np.ones(len(crops)), np.zeros(len(others))],
                               np.r_[crops, others]))


def weeds_caught(p_crop: np.ndarray, crops: np.ndarray, weeds: np.ndarray,
                 crops_lost: float) -> float | None:
    """The share of weeds under the P(crop) threshold that loses at most `crops_lost` of the
    crops. It doesn't depend on the rule file's species or on where 0.5 falls."""
    if not crops.any() or not weeds.any():
        return None
    ranked = np.sort(p_crop[crops])
    threshold = ranked[min(int(crops_lost * len(ranked) + 1e-9), len(ranked) - 1)]
    return float((p_crop[weeds] < threshold).mean())


def score_case(plants: Plants, base: np.ndarray, p_crop: np.ndarray, removal: np.ndarray,
               dataset: str, settings: dict[str, Any]) -> dict[str, Any]:
    """One way of combining, on a dataset's labelled plants: `p_crop` against the classifier's
    `base`, with `removable` of the probabilities it makes."""
    mask = plants.datasets == dataset
    crops, weeds = mask & (plants.labels == CROP), mask & (plants.labels == WEED)
    harmful = mask & plants.harmful
    fixed = ((base[crops] < 0.5) & (p_crop[crops] >= 0.5)).sum() + (
        (base[weeds] >= 0.5) & (p_crop[weeds] < 0.5)).sum()
    caused = ((base[crops] >= 0.5) & (p_crop[crops] < 0.5)).sum() + (
        (base[weeds] < 0.5) & (p_crop[weeds] >= 0.5)).sum()
    row = {"crops": int(crops.sum()), "weeds": int(weeds.sum()),
           "auc": _auc(p_crop[crops], p_crop[weeds]),
           "crops_wrong": float((p_crop[crops] < 0.5).mean()) if crops.any() else None,
           "weeds_wrong": float((p_crop[weeds] >= 0.5).mean()) if weeds.any() else None,
           "fixed": int(fixed), "caused": int(caused)}
    for target in settings["crops_lost"]:
        row[f"caught_at_{target}"] = weeds_caught(p_crop, crops, weeds, target)
        row[f"harmful_at_{target}"] = harmful_removed_at(
            _known(removal[crops]), _known(removal[harmful]), target)[0] if crops.any() else None
    return row


def compare_cases(plants: Plants, positions: np.ndarray, settings: dict[str, Any],
                  reference: tuple[Plants, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """The classifier alone and every way of adding a position, for each entry of
    settings["learn_from"] and dataset. `reference` (other plants and their positions) replaces
    other_datasets, for plants with nothing else to learn from.

    "classifier, stack" is the control: a stack with no position at all, so all it can do is
    recalibrate the classifier. What a stack with a position gains beyond it is the position's;
    the rest, such as harmful removed moving because remove now beats keep more often, isn't."""
    p_base = plants.p_crop
    base = logit(p_base)
    found = []
    datasets = np.unique(plants.datasets).tolist()
    removal = removable(plants, plants.probs)
    for dataset in datasets:
        found.append({"learn_from": "", "case": CLASSIFIER, "dataset": dataset,
                      "weight_classifier": None, "weight_position": None,
                      **score_case(plants, p_base, p_base, removal, dataset, settings)})
    for fold in settings["learn_from"]:
        for variant in (CLASSIFIER, *settings["variants"]):
            cells = table_cells(positions, variant, settings["cells"], settings["min_row_plants"])
            if fold == "other_datasets" and reference is not None:
                other = table_cells(reference[1], variant, settings["cells"],
                                    settings["min_row_plants"])
                parts = [(np.ones(len(base), bool),
                          learned_from(reference[0], other, logit(reference[0].p_crop)))]
            else:
                parts = [(judged, learned_from(plants, cells, base, learned))
                         for judged, learned in fold_masks(plants, fold)]
            ratios, stack, weights = combine(base, cells, parts, settings)
            cases = {} if variant == CLASSIFIER else {
                f"{variant} x{trust:g}": expit(base + trust * ratios)
                for trust in settings["trust"]}
            cases[f"{variant}, stack"] = expit(stack)
            for case, p_crop in cases.items():
                weight = (None, None)
                if case.endswith("stack") and weights is not None:
                    weight = weights
                removal = removable(plants, rescaled(plants, p_crop))
                for dataset in datasets:
                    found.append({"learn_from": fold, "case": case, "dataset": dataset,
                                  "weight_classifier": weight[0], "weight_position": weight[1],
                                  **score_case(plants, p_base, p_crop, removal, dataset, settings)})
    return found


def where_errors_are(plants: Plants, positions: np.ndarray,
                     settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Per dataset, in images with a real row: what a position could put right (a wrong crop on
    a row, a wrong weed off it), and what it puts at risk (a right crop off a row, a right weed
    on it). Plants without a real row are out of its reach either way."""
    real = positions["strongest_row"] >= settings["min_row_plants"]
    on_row = positions["row_distance"] < settings["on_row_mm"]
    p_crop = plants.p_crop
    found = []
    for dataset in np.unique(plants.datasets).tolist():
        mask = plants.datasets == dataset
        crops = mask & real & (plants.labels == CROP)
        weeds = mask & real & (plants.labels == WEED)
        wrong_crops, wrong_weeds = crops & (p_crop < 0.5), weeds & (p_crop >= 0.5)
        found.append({
            "dataset": dataset, "with_real_row": float(real[mask].mean()) if mask.any() else None,
            "crops": int(crops.sum()), "crops_wrong": int(wrong_crops.sum()),
            "crops_wrong_on_row": int((wrong_crops & on_row).sum()),
            "weeds": int(weeds.sum()), "weeds_wrong": int(wrong_weeds.sum()),
            "weeds_wrong_off_row": int((wrong_weeds & ~on_row).sum()),
            "could_fix": int((wrong_crops & on_row).sum() + (wrong_weeds & ~on_row).sum()),
            "at_risk": int((crops & ~on_row & (p_crop >= 0.5)).sum()
                           + (weeds & on_row & (p_crop < 0.5)).sum())})
    return found


def veto_cases(plants: Plants, positions: np.ndarray,
               settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Position only as a brake: a plant on a real row isn't removed, whatever the classifier
    says, so it can't cause a removal. At the rule file's defer_below, and at the same crops
    removed."""
    real = positions["strongest_row"] >= settings["min_row_plants"]
    on_row = real & (positions["row_distance"] < settings["on_row_mm"])
    veto = settings["veto"]
    spaced = (positions["spacing"] >= veto["spacing_above"]) & (
        np.isnan(positions["phase"]) | (positions["phase"] < veto["phase_below"]))
    removal = removable(plants, plants.probs)
    defer_below = plants.rules.defer_below or 0
    found = []
    for dataset in np.unique(plants.datasets).tolist():
        mask = plants.datasets == dataset
        crops, harmful = mask & (plants.labels == CROP), mask & plants.harmful
        for case, stopped in (("no veto", np.zeros(len(mask), bool)), ("on a row", on_row),
                              ("on a row, well spaced", on_row & spaced)):
            allowed = np.where(stopped, np.nan, removal)
            done = allowed >= defer_below
            row = {"dataset": dataset, "case": case,
                   "crops_removed": float(done[crops].mean()) if crops.any() else None,
                   "harmful_removed": float(done[harmful].mean()) if harmful.any() else None}
            for target in settings["crops_lost"]:
                row[f"harmful_at_{target}"] = None if not crops.any() else harmful_removed_at(
                    _known(allowed[crops]), _known(allowed[harmful]), target)[0]
            found.append(row)
    return found


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def position_alone(plants: Plants, positions: np.ndarray, geometries: dict[str, Geometry],
                   crop_of: dict[str, str], near: float) -> list[dict[str, Any]]:
    """How much the position says with no classifier, per dataset and session's main crop
    (`crop_of`); on a row means within `near` mm. Sessions without a labelled crop have no rows
    and aren't counted."""
    groups = defaultdict(list)
    for session in geometries:
        groups[session.split("/")[0], crop_of.get(session, "-")].append(session)
    found = []
    for (dataset, crop), sessions in sorted(groups.items()):
        mask = np.isin(plants.sessions, sessions)
        crops, weeds = mask & (plants.labels == CROP), mask & (plants.labels == WEED)
        distance, spacing, phase = (positions[name]
                                    for name in ("row_distance", "spacing", "phase"))
        on_row = distance < near
        chosen = [geometries[session] for session in sessions]
        found.append({
            "dataset": dataset, "crop": crop, "sessions": len(sessions),
            "periodic_sessions": sum(g.pitch is not None for g in chosen),
            "crops": int(crops.sum()), "weeds": int(weeds.sum()),
            "row_spacing": _median([g.row_spacing for g in chosen if g.row_spacing]),
            "usual_gap": _median([g.usual_gap for g in chosen if g.usual_gap]),
            "pitch": _median([g.pitch for g in chosen if g.pitch]),
            "crops_on_row": float(on_row[crops & ~np.isnan(distance)].mean())
            if (crops & ~np.isnan(distance)).any() else None,
            "weeds_on_row": float(on_row[weeds & ~np.isnan(distance)].mean())
            if (weeds & ~np.isnan(distance)).any() else None,
            "row_auc": _auc(-distance[crops], -distance[weeds]),
            "weeds_in_row": int((weeds & on_row).sum()),
            "spacing_auc": _auc(spacing[crops & on_row], spacing[weeds & on_row]),
            "phase_auc": _auc(-phase[crops & on_row], -phase[weeds & on_row])})
    return found


def main_crops(gt: dict[str, Any]) -> dict[str, str]:
    """The most often labelled crop of each session."""
    names = {c["id"]: c["name"] for c in gt["categories"]}
    tags = {c["id"]: c.get("supercategory") for c in gt["categories"]}
    images = {image["id"]: image for image in gt["images"]}
    counts = defaultdict(Counter)
    for annotation in gt["annotations"]:
        if tags[annotation["category_id"]] == "crop":
            counts[_session(images[annotation["image_id"]])][names[annotation["category_id"]]] += 1
    return {session: found.most_common(1)[0][0] for session, found in counts.items()}


def _number(values: list[Any], digits: int = 3) -> str:
    known = [value for value in values if value not in (None, "")]
    if not known:
        return "-"
    low, high = f"{min(known):.{digits}f}", f"{max(known):.{digits}f}"
    return low if low == high else f"{low}-{high}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|---" * len(headers) + "|",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _by(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    """The rows per value of `keys`, in the order first seen."""
    found = defaultdict(list)
    for row in rows:
        found[tuple(row[key] for key in keys)].append(row)
    return found


def _alone_section(rows: list[dict[str, Any]]) -> list[str]:
    def millimetres(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"

    intro = ("Rows and spacing from the other labelled crops; no classifier. On a row: within "
             "the config's on_row_mm. AUC: how often a crop looks better than a weed, by distance "
             "to the row, and for those on a row by distance to the next crop (spacing) and to an "
             "expected plant (phase).")
    return ["## Position alone", "", intro, "",
            *_table(["dataset", "crop", "sessions", "periodic", "crops", "weeds", "rows mm",
                     "gap mm", "pitch mm", "crops on row", "weeds on row", "row AUC",
                     "weeds in row", "spacing AUC", "phase AUC"],
                    [[row["dataset"], row["crop"], str(row["sessions"]),
                      str(row["periodic_sessions"]), str(row["crops"]), str(row["weeds"]),
                      millimetres(row["row_spacing"]), millimetres(row["usual_gap"]),
                      millimetres(row["pitch"]), spread([row["crops_on_row"]]),
                      spread([row["weeds_on_row"]]), _number([row["row_auc"]]),
                      str(row["weeds_in_row"]), _number([row["spacing_auc"]]),
                      _number([row["phase_auc"]])] for row in rows])]


def _of_group(rows: list[dict[str, Any]], group: str, source: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["group"] == group and row["evidence"] == source]


def _plants_section(title: str, intro: str, results: dict[str, list[dict[str, Any]]],
                    settings: dict[str, Any], group: str) -> list[str]:
    """One group of runs: where the errors are, each case, and the veto; a range is its runs."""
    lines = [f"## {title}", "", intro, ""]
    targets = settings["crops_lost"]
    for source in settings["evidence"]:
        where = _of_group(results["where"], group, source)
        if not where:
            continue
        about = ("Where the errors are, in images with a real row. Could fix: wrong crops on a "
                 "row and wrong weeds off it. At risk: right crops off a row and right weeds on it.")
        lines += [f"### Rows from: {source}", "", about, "",
                  *_table(["dataset", "plants with a real row", "crops", "wrong", "wrong, on row",
                           "weeds", "wrong", "wrong, off row", "could fix", "at risk"],
                          [[dataset, spread([r["with_real_row"] for r in rows]),
                            *(_number([r[key] for r in rows], 0) for key in (
                                "crops", "crops_wrong", "crops_wrong_on_row", "weeds",
                                "weeds_wrong", "weeds_wrong_off_row", "could_fix", "at_risk"))]
                           for (dataset,), rows in _by(where, "dataset").items()])]
        cases = _of_group(results["cases"], group, source)
        for dataset, rows in sorted(_by(cases, "dataset").items()):
            headers = ["learned from", "case", "AUC", "crops under 0.5", "weeds at 0.5 or more",
                       "fixed", "caused", *(f"weeds caught, <={t:.0%} crops" for t in targets),
                       *(f"harmful, <={t:.0%} crops" for t in targets), "weights"]
            table = []
            for (learn_from, case), found in _by(rows, "learn_from", "case").items():
                table.append([
                    learn_from or "-", case, _number([r["auc"] for r in found]),
                    spread([r["crops_wrong"] for r in found]),
                    spread([r["weeds_wrong"] for r in found]),
                    _number([r["fixed"] for r in found], 0),
                    _number([r["caused"] for r in found], 0),
                    *(spread([r[f"caught_at_{t}"] for r in found]) for t in targets),
                    *(spread([r[f"harmful_at_{t}"] for r in found]) for t in targets),
                    " / ".join(_number([r[key] for r in found], 2) for key in (
                        "weight_classifier", "weight_position"))
                    if found[0]["weight_classifier"] is not None else "-"])
            lines += [f"{dataset[0]}: {rows[0]['crops']} crops, {rows[0]['weeds']} weeds.", "",
                      *_table(headers, table)]
        veto = _of_group(results["veto"], group, source)
        lines += ["A brake only: a plant on a real row isn't removed.", "",
                  *_table(["dataset", "case", "crops removed", "harmful removed",
                           *(f"harmful, <={t:.0%} crops" for t in targets)],
                          [[dataset, case, *(spread([r[key] for r in rows]) for key in (
                              "crops_removed", "harmful_removed",
                              *(f"harmful_at_{t}" for t in targets)))]
                           for (dataset, case), rows in _by(veto, "dataset", "case").items()])]
    return lines


def _evaluate(plants: Plants, group: str, run: str, settings: dict[str, Any],
              results: dict[str, list[dict[str, Any]]], rows: RowSettings,
              references: dict[str, tuple[Plants, np.ndarray]] | None = None,
              keep: dict[str, tuple[Plants, np.ndarray]] | None = None) -> None:
    """Add a run's numbers to `results`, for each source of rows."""
    tag = {"group": group, "run": run}
    for source in settings["evidence"]:
        positions, _ = judge_positions(plants, source, rows)
        if keep is not None:
            keep.setdefault(source, (plants, positions))
        reference = references.get(source) if references else None
        labelled = {**tag, "evidence": source}
        results["where"] += [{**labelled, **r}
                             for r in where_errors_are(plants, positions, settings)]
        results["cases"] += [{**labelled, **r} for r in compare_cases(plants, positions, settings,
                                                                     reference)]
        results["veto"] += [{**labelled, **r} for r in veto_cases(plants, positions, settings)]


def write_report(settings: dict[str, Any], stem: Path) -> str:
    """Runs the three sets of plants (see module doc). Writes <stem>.md, <stem>.csv (every case
    per run), <stem>_where.csv, <stem>_veto.csv, <stem>_alone.csv and <stem>_sessions.csv (each
    session's rows, to trace a result); returns the Markdown."""
    rows = RowSettings(**{key: tuple(value) if isinstance(value, list) else value
                          for key, value in settings["rows"].items()})
    results = {"where": [], "cases": [], "veto": []}
    unseen = settings["unseen"]
    gt = load_coco(resolve(unseen["data_dir"]) / f"{settings['split']}.json")

    labelled = labelled_plants(gt)
    positions, geometries = judge_positions(labelled, "labelled", rows)
    crops = main_crops(gt)
    alone = position_alone(labelled, positions, geometries, crops, settings["on_row_mm"])
    sessions = [{"session": name, "crop": crops.get(name, "-"), "row_spacing": g.row_spacing,
                 "usual_gap": g.usual_gap, "pitch": g.pitch, "regularity": g.regularity,
                 "gaps": g.gaps} for name, g in sorted(geometries.items())]

    ground_truths, references = {}, {}
    for group, entries in settings["runs"].items():
        for entry in entries:
            run = run_dir(entry)
            folder = twostage_eval(resolve(settings["twostage_dir"]), run, settings["split"],
                                   settings["checkpoint"])
            if folder is None:
                raise FileNotFoundError(f"No two-stage evaluation of {run.name}, "
                                        f"{settings['split']} {settings['checkpoint']}; "
                                        "run scripts/predict_twostage.py")
            _evaluate(twostage_plants(folder, ground_truths), group, run.name, settings, results,
                      rows, keep=references)
    lines = [*_alone_section(alone)]
    for group, entries in settings["runs"].items():
        lines += _plants_section(
            f"Two-stage detections: {group}",
            f"Runs: {', '.join(run_dir(entry).name for entry in entries)}. Percentages of the "
            f"plants labelled, found by the detector; a range is the spread across runs. A stack's "
            f"0.5 is its own, so compare it by weeds caught.", results, settings, group)

    taxa = load_taxa(resolve(unseen["eppo_taxa"]))
    rules = load_rules(resolve(unseen["rules"]), taxa)
    missing = []
    for group, entries in unseen["runs"].items():
        chosen = []
        for entry in entries:
            run = run_dir(entry)
            path = resolve(unseen["probabilities"]) / run.name / (
                f"{settings['split']}_{settings['checkpoint']}.npz")
            if not path.exists():
                missing.append(run.name)
                continue
            chosen.append(run.name)
            plants = unseen_plants(path, gt, unseen["datasets"], rules, taxa)
            _evaluate(plants, f"unseen {group}", run.name, settings, results, rows, references)
        if chosen:
            lines += _plants_section(
                f"Labelled boxes, {group}: {', '.join(unseen['datasets'])}",
                f"Runs: {', '.join(chosen)}. Each dataset's own plants, with a perfect detector. "
                f"other_datasets here means the first two-stage run's plants.", results, settings,
                f"unseen {group}")
    if missing:
        about = (f"No saved probabilities for these runs, so the test of a new site is left out "
                 f"for them: {', '.join(missing)}. Run scripts/evaluate_unseen.py, here or on the "
                 f"training machine, and copy {unseen['probabilities']}/<run>/ here.")
        lines += ["## Not yet scored", "", about, ""]
    text = "\n".join(lines)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), results["cases"])
    for name, found in (("where", results["where"]), ("veto", results["veto"]), ("alone", alone),
                        ("sessions", sessions)):
        write_csv(stem.parent / f"{stem.name}_{name}.csv", found)
    return text
