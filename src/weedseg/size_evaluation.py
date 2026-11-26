"""Does a plant's real size help tell its species, once the classifier has had its say? Measured on
saved probabilities: no GPU and no retraining.

The classifier can't see a plant's size: every crop is cut around its box and resized to the input
size (weedseg.classifier.crops). The size is known in training (the box times the image's mm/px),
and a deployed model could have it from the camera, or from an estimate of the ground resolution.
This asks what it would be worth, with a stack on the classifier's log probabilities that is
learned on plants other than those judged (FOLDS):

- `control`: a temperature and a bias per class. A stack recalibrates by itself, and that moves
  decisions; the control shows by how much.
- `size`: the control plus, per class, b z + c z^2 of z, the standardised log of the box's longer
  side in mm: a log-normal size per species, on top of what the crop shows.
- `camera`: the same of the image's log mm/px instead of the plant's size. It knows the camera
  and nothing of the plant, so what it gains is the dataset: a check on what `size` gains.

An estimated ground resolution comes with an error. Each of `gsd_errors` (sigma) multiplies every
image's mm/px, and so each size in it, by exp(sigma x a standard normal draw), one draw per image.
The stack is learned on the true sizes and used on the wrong ones (`learned on true`), or learned
on wrong ones too (`learned with error`), as a model trained with that much size jitter would be.

Folds: `other_sessions` splits the sessions (dataset and group) into `session_parts` parts with
about as many plants each, and judges each part with a stack learned on the others;
`other_datasets` judges each dataset with a stack learned on the rest, the test of a new place.
Only classes with at least `min_fit_plants` labelled plants among those learned from get a bias
or a size term; the others keep what the classifier gave them, at the stack's temperature.

The plants are scripts/experiments/evaluate_tta.py's plain view, kept as weedseg.tta keeps them: the labelled
boxes of the crop cache (all datasets) and the two-stage evaluation's detections (datasets with
masks). scripts/experiments/evaluate_size.py runs it. See docs/twostage.md (Plant size).
"""

import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from weedseg.classifier.crops import read_index
from weedseg.compare import run_dir, twostage_eval
from weedseg.paths import resolve
from weedseg.skew_evaluation import (
    ALL,
    SHARES,
    Evaluation,
    Session,
    load_evaluation,
    scores,
    spread,
    write_csv,
)
from weedseg.tta import (
    COUNTS,
    PLAIN,
    PLANTS_FROM,
    as_saved,
    collect_views,
    crop_evaluation,
    partial_fit,
)

FOLDS = {"other_sessions": "other sessions", "other_datasets": "other datasets"}
CLASSIFIER = "classifier"
FEATURES = ("control", "size", "camera")
LEARNED = {"true": "learned on true", "error": "learned with error"}
LOG_FLOOR = 1e-7  # probabilities under it count as it, in logs and in the log loss


@dataclass
class Plants:
    """One set of plants: their evaluation (classes, rules, labels, sessions), the plain view's raw
    probabilities, and each plant's size in mm and its image's mm/px (NaN when unknown)."""

    evaluation: Evaluation
    probs: np.ndarray
    size_mm: np.ndarray
    mm_per_px: np.ndarray
    image_ids: np.ndarray
    rows: list[dict[str, Any]] | None = None  # crop index rows, for the partial-label fit

    @property
    def datasets(self) -> np.ndarray:
        return np.array([session[0] for session in self.evaluation.sessions])


def crop_plants(evaluation: Evaluation, rows: list[dict[str, Any]], probs: np.ndarray,
                groups: dict[int, str]) -> Plants:
    """The labelled boxes of the crop index `rows`, each in its image's session (`groups`: the
    group of every image of the split, those labelled with boxes only included)."""
    crops = crop_evaluation(evaluation, rows)
    crops = replace(crops, sessions=[(row["dataset"], groups.get(row["image_id"], "(none)"))
                                     for row in rows])
    nan = float("nan")
    return Plants(crops, probs, np.array([row["size_mm"] or nan for row in rows]),
                  np.array([row["mm_per_px"] or nan for row in rows]),
                  np.array([row["image_id"] for row in rows]), rows)


def detection_plants(evaluation: Evaluation, probs: np.ndarray) -> Plants:
    """The detections of a two-stage evaluation, sized by their own boxes."""
    images = {image["id"]: image for image in evaluation.gt["images"]}
    mm = np.array([images[i].get("mm_per_px") or float("nan") for i in evaluation.image_ids])
    longer = np.array([max(box[2], box[3]) for box in evaluation.boxes], dtype=np.float64)
    return Plants(evaluation, probs, longer * mm, mm, np.array(evaluation.image_ids))


def session_parts(sessions: list[Session], parts: int) -> np.ndarray:
    """The part of each plant: whole sessions, the largest first, each into the part with the
    fewest plants so far."""
    load = [0] * parts
    part_of = {}
    for session, count in sorted(Counter(sessions).items(), key=lambda item: (-item[1], item[0])):
        part = int(np.argmin(load))
        part_of[session] = part
        load[part] += count
    return np.array([part_of[session] for session in sessions])


def fold_masks(plants: Plants, fold: str, parts: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(plants judged, plants learned from) for every part of `fold`. The two never share a
    session, and for other_datasets never a dataset."""
    if fold == "other_datasets":
        datasets = plants.datasets
        return [(datasets == name, datasets != name) for name in np.unique(datasets)]
    if fold != "other_sessions":
        raise ValueError(f"fold must be one of {list(FOLDS)}, not {fold!r}")
    part = session_parts(plants.evaluation.sessions, parts)
    return [(part == k, part != k) for k in range(parts) if (part == k).any()]


def with_error(values: np.ndarray, image_ids: np.ndarray, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """`values` times exp(sigma x a standard normal draw), one draw per image."""
    images, position = np.unique(image_ids, return_inverse=True)
    return values * np.exp(sigma * rng.standard_normal(len(images)))[position]


@dataclass
class Stack:
    """logits = temperature x log p + bias + [z, z^2] @ terms, z = (log value - mean) / std."""

    temperature: float
    bias: np.ndarray  # [classes]
    terms: np.ndarray  # [0 or 2, classes]
    mean: float = 0.0
    std: float = 1.0

    def features(self, values: np.ndarray | None, count: int) -> np.ndarray:
        """[count, 0] without values, else [count, 2]: z and z^2, 0 (the mean) where unknown."""
        if values is None:
            return np.zeros((count, 0))
        logs = np.log(values)
        z = np.where(np.isfinite(logs), (logs - self.mean) / self.std, 0.0)
        return np.stack([z, z ** 2], 1)

    def probs(self, log_probs: np.ndarray, values: np.ndarray | None) -> np.ndarray:
        logits = (self.temperature * log_probs + self.bias
                  + self.features(values, len(log_probs)) @ self.terms)
        logits -= logits.max(1, keepdims=True)
        found = np.exp(logits)
        return (found / found.sum(1, keepdims=True)).astype(np.float32)


def fit_stack(log_probs: np.ndarray, true: np.ndarray, values: np.ndarray | None,
              settings: dict[str, Any]) -> Stack:
    """A stack learned on labelled plants (multinomial cross-entropy, L2 `l2` on the bias and the
    size terms, not on the temperature). Classes with under `min_fit_plants` plants get neither."""
    classes = log_probs.shape[1]
    fitted = np.bincount(true, minlength=classes) >= settings["min_fit_plants"]
    stack = Stack(1.0, np.zeros(classes), np.zeros((0 if values is None else 2, classes)))
    if values is not None:
        logs = np.log(values)
        logs = logs[np.isfinite(logs)]
        stack.mean, stack.std = float(logs.mean()), float(logs.std()) or 1.0
    x = torch.from_numpy(log_probs.astype(np.float64))
    y = torch.from_numpy(true.astype(np.int64))
    extra = torch.from_numpy(stack.features(values, len(log_probs)))
    mask = torch.from_numpy(fitted.astype(np.float64))
    temperature = torch.ones((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    terms = torch.zeros(stack.terms.shape, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([temperature, bias, terms], lr=1, max_iter=settings["max_iter"],
                                  tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20,
                                  line_search_fn="strong_wolfe")

    def loss() -> torch.Tensor:
        optimiser.zero_grad()
        b, t = bias * mask, terms * mask
        value = (F.cross_entropy(temperature * x + b + extra @ t, y)
                 + settings["l2"] * (b.square().sum() + t.square().sum()))
        value.backward()
        return value

    optimiser.step(loss)
    with torch.no_grad():
        return replace(stack, temperature=float(temperature), bias=(bias * mask).numpy(),
                       terms=(terms * mask).numpy())


def stacked(plants: Plants, fold: str, settings: dict[str, Any], learn: np.ndarray | None,
            judge: list[np.ndarray | None]) -> list[np.ndarray]:
    """Probabilities [plants, classes] for each of the `judge` values: every plant judged by a
    stack learned with `learn` on the labelled plants of the other parts of `fold`."""
    log_probs = np.log(np.clip(plants.probs, LOG_FLOOR, None)).astype(np.float64)
    true = plants.evaluation.true
    found = [np.zeros(log_probs.shape, np.float32) for _ in judge]
    for judged, learned in fold_masks(plants, fold, settings["session_parts"]):
        fit = learned & (true >= 0)
        if learn is not None:
            fit &= np.isfinite(np.log(learn))
        stack = fit_stack(log_probs[fit], true[fit], None if learn is None else learn[fit],
                          settings)
        for probs, values in zip(found, judge):
            probs[judged] = stack.probs(log_probs[judged],
                                        None if values is None else values[judged])
    return found


def cases(plants: Plants, settings: dict[str, Any]) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    """(what it is, probabilities) for the classifier and every stack, one at a time: each holds
    as much memory as the plain view."""
    yield ({"variant": CLASSIFIER, "fold": None, "error": None, "learned": None}, plants.probs)
    errors = settings["gsd_errors"]
    wrong = {sigma: with_error(plants.size_mm, plants.image_ids, sigma,
                               np.random.default_rng([settings["seed"], round(sigma * 1000)]))
             for sigma in errors}
    for fold in settings["folds"]:
        base = {"fold": fold, "error": None, "learned": None}
        yield {**base, "variant": "control"}, stacked(plants, fold, settings, None, [None])[0]
        yield ({**base, "variant": "camera"},
               stacked(plants, fold, settings, plants.mm_per_px, [plants.mm_per_px])[0])
        size, *on_wrong = stacked(plants, fold, settings, plants.size_mm,
                                  [plants.size_mm, *wrong.values()])
        yield {**base, "variant": "size"}, size
        for sigma, probs in zip(errors, on_wrong):
            yield {**base, "variant": "size", "error": sigma, "learned": "true"}, probs
        del size, on_wrong
        for sigma in errors:
            probs = stacked(plants, fold, settings, wrong[sigma], [wrong[sigma]])[0]
            yield {**base, "variant": "size", "error": sigma, "learned": "error"}, probs


def log_loss(probs: np.ndarray, true: np.ndarray) -> float | None:
    """Mean -log p(true class) of the labelled plants."""
    if not len(true):
        return None
    return float(-np.log(np.clip(probs[np.arange(len(true)), true], LOG_FLOOR, None)).mean())


def score_cases(plants: Plants, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Per case and dataset (and ALL): `scores`, the log loss, the partial-label fit (crops),
    the median size, and what the case did to the labelled plants' top classes against the
    classifier."""
    labelled = np.flatnonzero(plants.evaluation.true >= 0)
    true = plants.evaluation.true[labelled]
    datasets = plants.datasets[labelled]
    before = as_saved(plants.probs)[labelled].argmax(1)
    found = []
    for case, probs in cases(plants, settings):
        saved = as_saved(probs)
        after = saved[labelled].argmax(1)
        fits = partial_fit(plants.rows, saved, plants.evaluation.classes) if plants.rows else {}
        for group, values in scores(plants.evaluation, saved, settings).items():
            mask = np.ones(len(labelled), bool) if group == ALL else datasets == group
            was_right, is_right = before[mask] == true[mask], after[mask] == true[mask]
            sizes = plants.size_mm[labelled][mask]
            sizes = sizes[np.isfinite(sizes)]
            found.append({"dataset": group, **case, **values,
                          "log_loss": log_loss(probs[labelled][mask], true[mask]),
                          "partial_fit": fits.get(group),
                          "size_median_mm": float(np.median(sizes)) if len(sizes) else None,
                          "wrong_to_right": int((~was_right & is_right).sum()),
                          "right_to_wrong": int((was_right & ~is_right).sum())})
    return found


def plain_view(folder: Path, ids: np.ndarray, classes: list[str]) -> np.ndarray:
    """The plain view's probabilities kept in `folder` by scripts/experiments/evaluate_tta.py, in the order of
    `classes`."""
    def missing(view: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        raise SystemExit(f"No plain view of these plants in {folder}: run scripts/experiments/evaluate_tta.py "
                         "on this run first (max_plants null)")
    return collect_views(folder, ids, {PLAIN: {}}, missing, classes)[PLAIN]


def split_groups(evaluation_folder: Path) -> dict[int, str]:
    """The group of every image of the evaluation's split file, those labelled with boxes only
    included (the two-stage ground truth leaves them out)."""
    evaluation = json.loads((evaluation_folder / "evaluation.json").read_text(encoding="utf-8"))
    with open(resolve(evaluation["split_file"]), encoding="utf-8") as f:
        images = json.load(f)["images"]
    return {image["id"]: image.get("group") or "(none)" for image in images}


def evaluate_run(run: Path, settings: dict[str, Any], ground_truths: dict[Path, dict],
                 groups: dict[Path, dict[int, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows of `score_cases` for both sets of plants of one classifier run, and lines checking its
    plain view against its normal evaluations."""
    split, checkpoint = settings["split"], settings["checkpoint"]
    folder = twostage_eval(resolve(settings["twostage_dir"]), run, split, checkpoint)
    if folder is None:
        raise SystemExit(f"No two-stage evaluation of {run.name}, {split} {checkpoint}")
    evaluation = load_evaluation(folder, ground_truths)
    split_file = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))["split_file"]
    if split_file not in groups:
        groups[split_file] = split_groups(folder)
    kept = resolve(settings["tta_dir"]) / run.name / f"{split}_{checkpoint}"
    rows = read_index(kept / "crops" / "plants.csv")
    crop_probs = plain_view(kept / "crops", np.array([row["annotation_id"] for row in rows]),
                            evaluation.classes)
    detection_ids = np.array([[image_id, *box]
                              for image_id, box in zip(evaluation.image_ids, evaluation.boxes)])
    detection_probs = plain_view(kept / "detections", detection_ids, evaluation.classes)
    sets = {"crops": crop_plants(evaluation, rows, crop_probs, groups[split_file]),
            "detections": detection_plants(evaluation, detection_probs)}
    found = []
    for kind, plants in sets.items():
        print(f"{run.name}, {kind}: {len(plants.probs)} plants, "
              f"{len(set(plants.evaluation.sessions))} sessions")
        found += [{"run": run.name, "plants_from": kind, **row}
                  for row in score_cases(plants, settings)]
    return found, plain_checks(run, evaluation, detection_probs, found, split, checkpoint)


def plain_checks(run: Path, evaluation: Evaluation, detection_probs: np.ndarray,
                 rows: list[dict[str, Any]], split: str, checkpoint: str) -> list[str]:
    """The classifier rows must give what the run's normal evaluations gave."""
    plain = as_saved(detection_probs)
    same = int((plain.argmax(1) == evaluation.probs.argmax(1)).sum())
    lines = [(f"{run.name}: detections, same top class as the two-stage evaluation on {same} of "
              f"{len(plain)}")]
    metrics = run / "eval" / f"{split}_{checkpoint}" / "metrics.json"
    if metrics.exists():
        accuracy = json.loads(metrics.read_text(encoding="utf-8"))["accuracy"]
        own = next(row for row in rows if row["plants_from"] == "crops"
                   and row["variant"] == CLASSIFIER and row["dataset"] == ALL)["class_right"]
        lines.append(f"{run.name}: labelled boxes, class right {own:.2%}, {metrics.name} "
                     f"accuracy {accuracy:.2%}")
    return lines


def number_spread(values: list[float | None], digits: int = 3) -> str:
    """Across runs: a number, or the range when the runs differ at this rounding."""
    known = [value for value in values if value is not None]
    if not known:
        return "-"
    low, high = f"{min(known):.{digits}f}", f"{max(known):.{digits}f}"
    return low if low == high else f"{low}-{high}"


def case_name(row: dict[str, Any]) -> str:
    if row["variant"] == CLASSIFIER:
        return CLASSIFIER
    name = f"{row['variant']}, {FOLDS[row['fold']]}"
    if row["error"] is not None:
        name += f", mm/px error ~{math.expm1(row['error']):.0%}, {LEARNED[row['learned']]}"
    return name


def _table(rows: list[dict[str, Any]], columns: dict[str, str], counts: bool) -> list[str]:
    titles = [*columns.values(), *(COUNTS.values() if counts else ())]
    lines = ["| case | " + " | ".join(titles) + " |", "|---" * (1 + len(titles)) + "|"]
    for name in dict.fromkeys(case_name(row) for row in rows):
        here = [row for row in rows if case_name(row) == name]
        cells = [number_spread([row[key] for row in here]) if key == "log_loss"
                 else spread([row[key] for row in here], 2) for key in columns]
        if counts:
            cells += [number_spread([row[key] for row in here], 0) for key in COUNTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def report(rows: list[dict[str, Any]], settings: dict[str, Any], runs: list[str],
           checks: list[str]) -> str:
    """The Markdown report: per set of plants and dataset, the classifier and the stacks without
    an error; then the stacks with an error in the ground resolution. Ranges are across runs."""
    targets = settings["crops_removed"]
    harmful = {f"harmful_at_{t}": f"harmful at <={t:.0%} crops" for t in targets}
    shares = {**SHARES, **harmful, "log_loss": "log loss", "partial_fit": "partial-label fit"}
    intro = ("Percentages of the labelled plants; a range is the spread across runs. Each stack "
             "judges a plant with what it learned on plants of other sessions or other datasets. "
             "control = temperature and class bias only; size = control + a log-normal size per "
             "class; camera = the same on the image's mm/px instead (a dataset check, not a "
             "size). Judge size against the control of the same fold, not against the "
             "classifier: a stack recalibrates by itself. Log loss: mean -log p(true class), "
             "lower is better. Wrong to right and right to wrong count top classes against the "
             "classifier. See docs/twostage.md (Plant size).")
    errors = ", ".join(f"~{math.expm1(sigma):.0%} (sigma {sigma:g})"
                       for sigma in settings["gsd_errors"])
    lines = [f"Runs: {', '.join(runs)}.", "", intro, "",
             (f"Errors in mm/px, one draw per image, as one standard deviation of the factor "
              f"exp(sigma x normal): {errors}. Learned on true: the stack learned on the true "
              "sizes; learned with error: on sizes with the same error."), "",
             "The classifier rows against the normal evaluations:", "",
             *(f"- {check}" for check in checks), ""]
    for kind, title in PLANTS_FROM.items():
        chosen = [row for row in rows if row["plants_from"] == kind]
        if not chosen:
            continue
        columns = {key: name for key, name in shares.items()
                   if any(row.get(key) is not None for row in chosen)}
        order = sorted({row["dataset"] for row in chosen}, key=lambda name: (name != ALL, name))
        lines += [f"# {title}", ""]
        for dataset in order:
            here = [row for row in chosen if row["dataset"] == dataset and row["error"] is None]
            size = here[0]["size_median_mm"]
            heading = f"## {dataset}: {here[0]['plants']} plants"
            if size:
                heading += f", median size {size:.0f} mm"
            lines += [heading, "", *_table(here, columns, True), ""]
        brief = {key: shares[key] for key in ("species_right", "balanced", *harmful, "log_loss")
                 if key in columns}
        lines += [f"## {title}: with an error in the ground resolution", ""]
        for dataset in order:
            here = [row for row in chosen if row["dataset"] == dataset
                    and (row["variant"] == "size" or row["variant"] == "control")]
            lines += [f"### {dataset}", "", *_table(here, brief, False), ""]
    return "\n".join(lines)


def write_report(settings: dict[str, Any], stem: Path) -> str:
    """Score every case of every run. Writes <stem>.md and <stem>.csv (every number per run);
    returns the Markdown."""
    torch.set_num_threads(settings["threads"])
    rows, checks, ground_truths, groups = [], [], {}, {}
    runs = [run_dir(entry) for entry in settings["runs"]]
    for run in runs:
        found, lines = evaluate_run(run, settings, ground_truths, groups)
        rows += found
        checks += lines
    text = report(rows, settings, [run.name for run in runs], checks)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    return text
