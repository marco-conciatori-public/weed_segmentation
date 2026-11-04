"""Test-time augmentation: does the classifier do better on several views of a plant than on one?

A view is something training has seen: a flip, quarter turns, a tighter or looser window. Every
view is predicted once for every plant and kept, so which views, how they are combined and which
plants get them are all worked out afterwards, without a GPU:

- `mean` averages the probabilities; `vote` gives each class its share of the views' top classes,
  ties going to the larger mean;
- a gate uses them only for plants whose plain view has a top probability under it. A gate re-rolls
  only the low scores, so some rise over a threshold without anything new being known: compare
  decisions at the same crops removed, and against the same views without a gate.

Plants come from two places, scored alike (weedseg.skew_evaluation.scores):

- `crops`: the labelled boxes of the crop cache, classifier alone, for all datasets;
- `detections`: the two-stage evaluation's detections, with their real boxes and the plant
  threshold, for the datasets that have masks.

Probabilities are combined first and then treated as the two-stage pipeline treats its own
(`as_saved`), so the plain view gives the numbers of a normal evaluation.

scripts/evaluate_tta.py runs it. See docs/twostage.md (Test-time augmentation).
"""

import json
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from weedseg.classifier.crops import cache_dir, read_index
from weedseg.classifier.data import CropDataset
from weedseg.classifier.engine import predict_probs
from weedseg.classifier.model import load_classifier
from weedseg.compare import run_dir, twostage_eval
from weedseg.maskrcnn.engine import CLASS_SCORE_MIN, select_device
from weedseg.paths import resolve
from weedseg.skew_evaluation import (
    ALL,
    SHARES,
    Evaluation,
    load_evaluation,
    scores,
    spread,
    write_csv,
)
from weedseg.twostage import image_probs

PLAIN = "plain"
VIEW_KEYS = {"flip", "rotate90", "scale"}
# In a vote, the part of a class's share that comes from its mean: enough to break a tie.
TIE_SHARE = 1e-3
PLANTS_FROM = {"crops": "Classifier alone, on the labelled boxes of all datasets",
               "detections": "Both stages, on the plants found (datasets with masks only)"}
COUNTS = {"wrong_to_right": "wrong to right", "right_to_wrong": "right to wrong"}


class Variant(NamedTuple):
    """A way to use the views: which ones, how they are combined, on which plants."""

    name: str
    views: list[str]
    method: str = "mean"
    gate: float | None = None


def check_views(settings: dict[str, Any]) -> None:
    """Raise ValueError if the views or the sets of views in `settings` can't work."""
    views, sets = settings["views"], settings["view_sets"]
    if views.get(PLAIN, None) != {}:
        raise ValueError(f"views must have {PLAIN}: {{}}, the crop as it is")
    for name, view in views.items():
        unknown = set(view) - VIEW_KEYS
        if unknown:
            raise ValueError(f"View {name} has unknown settings: {', '.join(sorted(unknown))}")
        if view.get("rotate90", 0) not in range(4):
            raise ValueError(f"View {name}: rotate90 is a number of quarter turns, 0 to 3")
        if view.get("scale", 1.0) <= 0:
            raise ValueError(f"View {name}: scale must be positive")
    for name, members in sets.items():
        unknown = set(members) - set(views)
        if unknown:
            raise ValueError(f"View set {name} names unknown views: {', '.join(sorted(unknown))}")
        if PLAIN not in members:
            raise ValueError(f"View set {name} must include {PLAIN}")
    if settings["main_set"] not in sets:
        raise ValueError(f"main_set {settings['main_set']} isn't in view_sets")


def check_scales(views: dict[str, dict[str, Any]], crops: dict[str, Any]) -> None:
    """A window larger than the crop cache holds would show the fill colour as a border."""
    for name, view in views.items():
        if view.get("scale", 1.0) > crops["cache_context"]:
            raise ValueError(f"View {name}: scale is over the cache's context, "
                             f"{crops['cache_context']:g} (crops.cache_context)")


def variants(settings: dict[str, Any]) -> list[Variant]:
    """The plain crop, each view set averaged, and the main set by vote and behind each gate."""
    main = settings["main_set"]
    members = settings["view_sets"][main]
    return [Variant(PLAIN, [PLAIN]),
            *(Variant(name, views) for name, views in settings["view_sets"].items()),
            *([Variant(f"{main}, votes", members, "vote")] if settings["votes"] else []),
            *(Variant(f"{main}, under {gate:g}", members, "mean", gate)
              for gate in settings["gates"])]


def aggregate(views: np.ndarray, method: str) -> np.ndarray:
    """[views, plants, classes] -> [plants, classes], as described at the top of this module."""
    mean = views.mean(0)
    if method == "mean":
        return mean
    if method != "vote":
        raise ValueError(f"method must be mean or vote, not {method!r}")
    votes = np.zeros_like(mean)
    for top in views.argmax(2):
        votes[np.arange(len(top)), top] += 1 / len(views)
    return (1 - TIE_SHARE) * votes + TIE_SHARE * mean


def gated(plain: np.ndarray, combined: np.ndarray, gate: float | None) -> np.ndarray:
    """`combined` for the plants whose plain top probability is under `gate`, else `plain`."""
    if gate is None:
        return combined
    return np.where(plain.max(1, keepdims=True) < gate, combined, plain)


def as_saved(probs: np.ndarray) -> np.ndarray:
    """Probabilities as the pipeline uses them: weedseg.twostage.with_species drops those under
    CLASS_SCORE_MIN and rounds the rest, and decisions then scale them to sum to 1."""
    kept = np.round(np.where(probs >= CLASS_SCORE_MIN, probs, 0.0), 4)
    return kept / kept.sum(1, keepdims=True)


def combine(variant: Variant, views: dict[str, np.ndarray]) -> np.ndarray:
    """A variant's probabilities [plants, classes]."""
    stack = np.stack([views[name] for name in variant.views])
    return as_saved(gated(views[PLAIN], aggregate(stack, variant.method), variant.gate))


def pick(total: int, max_plants: int | None) -> np.ndarray:
    """All positions, or a fixed random `max_plants` of them, in order (for a smoke test)."""
    if max_plants is None or max_plants >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(0).choice(total, max_plants, replace=False))


def subset_evaluation(evaluation: Evaluation, keep: np.ndarray) -> Evaluation:
    """An evaluation with only these detections."""
    return replace(evaluation, probs=evaluation.probs[keep],
                   plant_scores=evaluation.plant_scores[keep],
                   sessions=[evaluation.sessions[i] for i in keep], true=evaluation.true[keep],
                   image_ids=[evaluation.image_ids[i] for i in keep],
                   boxes=[evaluation.boxes[i] for i in keep])


def crop_evaluation(evaluation: Evaluation, rows: list[dict[str, Any]]) -> Evaluation:
    """`evaluation`'s classes and rules for the labelled boxes `rows` (crop index rows): each is
    a plant for sure, and a partial label has no class to be right about (-1)."""
    column = {name: k for k, name in enumerate(evaluation.classes)}
    return replace(evaluation, probs=np.zeros((len(rows), len(column))),
                   plant_scores=np.ones(len(rows)),
                   sessions=[(row["dataset"], "(none)") for row in rows],
                   true=np.array([-1 if row["allowed"] else column[row["category"]]
                                  for row in rows], dtype=np.int64),
                   image_ids=[row["image_id"] for row in rows], boxes=[])


def partial_fit(rows: list[dict[str, Any]], probs: np.ndarray,
                classes: list[str]) -> dict[str, float]:
    """Per dataset and ALL: the share of partly labelled plants whose top class their label
    allows."""
    fits = defaultdict(list)
    for row, top in zip(rows, probs.argmax(1)):
        if row["allowed"]:
            fit = classes[top] in row["allowed"]
            fits[row["dataset"]].append(fit)
            fits[ALL].append(fit)
    return {group: float(np.mean(values)) for group, values in fits.items()}


def evaluate_variants(evaluation: Evaluation, views: dict[str, np.ndarray],
                      settings: dict[str, Any],
                      rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per variant and dataset: `scores`, and what the variant did to the top classes of the
    labelled plants, against the plain view. `rows` (crop index rows) add the partial-label fit."""
    plants = np.flatnonzero(evaluation.true >= 0)
    true = evaluation.true[plants]
    before = as_saved(views[PLAIN])[plants].argmax(1)
    datasets = np.array([evaluation.sessions[row][0] for row in plants])
    found = []
    for variant in variants(settings):
        probs = combine(variant, views)
        after = probs[plants].argmax(1)
        fits = partial_fit(rows, probs, evaluation.classes) if rows else {}
        for group, values in scores(evaluation, probs, settings).items():
            mask = np.ones(len(plants), bool) if group == ALL else datasets == group
            was_right, is_right = before[mask] == true[mask], after[mask] == true[mask]
            found.append({"dataset": group, "variant": variant.name, **values,
                          "partial_fit": fits.get(group),
                          "wrong_to_right": int((~was_right & is_right).sum()),
                          "right_to_wrong": int((was_right & ~is_right).sum())})
    return found


def saved_view(path: Path, ids: np.ndarray,
               view: dict[str, Any]) -> tuple[np.ndarray, list[str]] | None:
    """The probabilities and classes kept at `path`, if they are of these plants and this view."""
    if not path.exists():
        return None
    with np.load(path) as saved:
        if (np.array_equal(saved["ids"], ids) and json.loads(str(saved["view"])) == view):
            return saved["probs"], saved["classes"].tolist()
    print(f"{path} holds other plants or another view: predicting again")
    return None


def save_view(path: Path, ids: np.ndarray, view: dict[str, Any], probs: np.ndarray,
              classes: list[str]) -> None:
    """Keep a view's probabilities; a session stopped while writing leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.npz")
    np.savez(partial, ids=ids, view=json.dumps(view), probs=probs, classes=np.array(classes))
    partial.replace(path)


def collect_views(folder: Path, ids: np.ndarray, views: dict[str, dict[str, Any]],
                  predict: Callable[[dict[str, Any]], tuple[np.ndarray, list[str]]],
                  classes: list[str]) -> dict[str, np.ndarray]:
    """Each view's probabilities [plants, classes], in the order of `classes`: kept in `folder`,
    or else `predict(view)` and kept."""
    found = {}
    for name, view in views.items():
        path = folder / f"{name}.npz"
        saved = saved_view(path, ids, view)
        if saved is None:
            print(f"{folder.name}, {name}: {len(ids)} plants")
            saved = predict(view)
            save_view(path, ids, view, *saved)
        probs, own = saved
        if set(own) != set(classes):
            raise SystemExit(f"{path} has other classes than the evaluation: "
                             f"{sorted(set(own) ^ set(classes))}")
        found[name] = probs[:, [own.index(k) for k in classes]]
    return found


def crop_rows(run_config: dict[str, Any], split: str, kept: Path) -> list[dict[str, Any]]:
    """The crop index of a split. A copy stays next to the views, so that the report can be made
    again where the crop cache isn't, e.g. on the laptop."""
    index = cache_dir(resolve(run_config["data"]["dir"]), run_config["crops"]) / f"index_{split}.csv"
    if index.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(index, kept)
    elif not kept.exists():
        raise SystemExit(f"No crop cache {index}; run scripts/prepare_crops.py first")
    return read_index(kept)


def crop_view(model: nn.Module, classes: list[str], rows: list[dict[str, Any]], folder: Path,
              crops: dict[str, Any], view: dict[str, Any], device: torch.device, amp: bool,
              batch_size: int, num_workers: int) -> np.ndarray:
    """Probabilities [rows, classes] of one view of cached crops."""
    dataset = CropDataset(rows, folder, classes, crops, view=view)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers)
    return predict_probs(model, loader, device, amp)[0]


def detection_view(model: nn.Module, classes: list[str], evaluation: Evaluation, image_dir: Path,
                   crops: dict[str, Any], view: dict[str, Any], device: torch.device, amp: bool,
                   batch_size: int, num_workers: int) -> np.ndarray:
    """Probabilities [detections, classes] of one view of an evaluation's detections, cut from
    the full images around their boxes."""
    by_image = defaultdict(list)
    for row, (image_id, box) in enumerate(zip(evaluation.image_ids, evaluation.boxes)):
        by_image[image_id].append({"row": row, "bbox": box})
    found = np.zeros((len(evaluation.boxes), len(classes)), np.float32)
    for image_id, probs in image_probs(by_image, evaluation.gt["images"], image_dir, model, crops,
                                       device, amp, batch_size, num_workers, view):
        found[[detection["row"] for detection in by_image[image_id]]] = probs
    return found


def evaluate_run(run: Path, settings: dict[str, Any], ground_truths: dict[Path, dict],
                 device: torch.device, out_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Predict what isn't kept yet of one classifier run. Returns its rows (`evaluate_variants`,
    both sets of plants) and lines checking the plain view against its normal evaluation."""
    split, checkpoint = settings["split"], settings["checkpoint"]
    run_config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    crops = run_config["crops"]
    check_scales(settings["views"], crops)
    if crops["background"] != "keep":
        raise SystemExit(f"{run.name}: only crops.background keep is supported here")
    folder = twostage_eval(resolve(settings["twostage_dir"]), run, split, checkpoint)
    if folder is None:
        raise SystemExit(f"No two-stage evaluation of {run.name}, {split} {checkpoint}; "
                         "run scripts/predict_twostage.py")
    evaluation = load_evaluation(folder, ground_truths)
    keep = pick(len(evaluation.boxes), settings["max_plants"])
    evaluation = subset_evaluation(evaluation, keep)
    base = out_dir / run.name / f"{split}_{checkpoint}"
    rows = crop_rows(run_config, split, base / "crops" / "plants.csv")
    rows = [rows[i] for i in pick(len(rows), settings["max_plants"])]
    amp = device.type == "cuda"

    @cache
    def classifier() -> tuple[nn.Module, list[str]]:
        """Loaded only if a view has to be predicted, so the report needs no checkpoint."""
        print(f"{run.name}: loading {checkpoint}.pt")
        model, loaded = load_classifier(run / f"{checkpoint}.pt", device)
        return model, loaded["classes"]  # last.pt holds the optimiser too: about 1 GB

    def predict_crop(view: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        model, classes = classifier()
        cached = cache_dir(resolve(run_config["data"]["dir"]), crops) / split
        return crop_view(model, classes, rows, cached, crops, view, device, amp,
                         settings["batch_size"], settings["num_workers"]), classes

    def predict_detection(view: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
        model, classes = classifier()
        image_dir = resolve(evaluation.gt["info"]["image_dir"])
        return detection_view(model, classes, evaluation, image_dir, crops, view, device, amp,
                              settings["batch_size"], settings["num_workers"]), classes

    crop_views = collect_views(base / "crops", np.array([r["annotation_id"] for r in rows]),
                               settings["views"], predict_crop, evaluation.classes)
    detection_ids = np.array([[image_id, *box] for image_id, box
                              in zip(evaluation.image_ids, evaluation.boxes)])
    detection_views = collect_views(base / "detections", detection_ids, settings["views"],
                                    predict_detection, evaluation.classes)

    found = []
    labelled_boxes = evaluate_variants(crop_evaluation(evaluation, rows), crop_views, settings, rows)
    found += [{"run": run.name, "plants_from": "crops", **row} for row in labelled_boxes]
    found += [{"run": run.name, "plants_from": "detections", **row}
              for row in evaluate_variants(evaluation, detection_views, settings)]
    return found, plain_checks(run, evaluation, detection_views[PLAIN], labelled_boxes, split,
                               checkpoint, settings["max_plants"])


def plain_checks(run: Path, evaluation: Evaluation, plain: np.ndarray,
                 labelled_boxes: list[dict[str, Any]], split: str, checkpoint: str,
                 max_plants: int | None) -> list[str]:
    """The plain view must give what the run's normal evaluations gave."""
    plain = as_saved(plain)
    same = int((plain.argmax(1) == evaluation.probs.argmax(1)).sum())
    gap = np.abs(plain - evaluation.probs).max()
    lines = [(f"{run.name}: detections, same top class as the two-stage evaluation on {same} of "
              f"{len(plain)}, largest probability difference {gap:.4f}")]
    metrics = run / "eval" / f"{split}_{checkpoint}" / "metrics.json"
    if metrics.exists() and max_plants is None:
        accuracy = json.loads(metrics.read_text(encoding="utf-8"))["accuracy"]
        own = next(row for row in labelled_boxes
                   if row["variant"] == PLAIN and row["dataset"] == ALL)["class_right"]
        lines.append(f"{run.name}: labelled boxes, class right {own:.2%}, {metrics.name} "
                     f"accuracy {accuracy:.2%}")
    return lines


def count_spread(values: list[int]) -> str:
    """Across runs: a count, or the range when the runs differ."""
    return str(values[0]) if min(values) == max(values) else f"{min(values)}-{max(values)}"


def report(rows: list[dict[str, Any]], settings: dict[str, Any], runs: list[str],
           checks: list[str]) -> str:
    """The Markdown report: per set of plants and dataset, one row per variant, with the range
    across runs."""
    targets = settings["crops_removed"]
    shares = {**SHARES, **{f"harmful_at_{t}": f"harmful at <={t:.0%} crops" for t in targets},
              "partial_fit": "partial-label fit"}
    intro = ("Percentages of the labelled plants; a range is the spread across runs, and the plain "
             "row's spread is the noise to compare with. Compare decisions at the same crops "
             "removed, not at the rule file's defer_below: combining moves the confidence. Wrong "
             "to right and right to wrong count top classes against the plain view. See "
             "docs/twostage.md (Test-time augmentation).")
    views = "; ".join(f"{name} = {' '.join(f'{k} {v}' for k, v in view.items()) or 'as it is'}"
                      for name, view in settings["views"].items())
    lines = [f"Runs: {', '.join(runs)}.", "", intro, "", f"Views: {views}.", "",
             "Plain view against the normal evaluations:", "",
             *(f"- {check}" for check in checks), ""]
    for kind, title in PLANTS_FROM.items():
        chosen = [row for row in rows if row["plants_from"] == kind]
        columns = {key: name for key, name in shares.items()
                   if any(row.get(key) is not None for row in chosen)}
        lines += [f"# {title}", ""]
        order = sorted({row["dataset"] for row in chosen}, key=lambda name: (name != ALL, name))
        for dataset in order:
            here = [row for row in chosen if row["dataset"] == dataset]
            lines += [f"## {dataset}: {here[0]['plants']} plants", "",
                      "| variant | " + " | ".join([*columns.values(), *COUNTS.values()]) + " |",
                      "|---" * (1 + len(columns) + len(COUNTS)) + "|"]
            for variant in dict.fromkeys(row["variant"] for row in here):
                cases = [row for row in here if row["variant"] == variant]
                cells = [spread([row[key] for row in cases], 2) for key in columns]
                cells += [count_spread([row[key] for row in cases]) for key in COUNTS]
                lines.append(f"| {variant} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def write_report(settings: dict[str, Any], stem: Path) -> str:
    """Predict what isn't kept yet and score every variant. Writes <stem>.md and <stem>.csv (every
    number per run); returns the Markdown."""
    check_views(settings)
    device = select_device(settings["device"])
    out_dir = resolve(settings["output_dir"])
    rows, checks, ground_truths = [], [], {}
    runs = [run_dir(entry) for entry in settings["runs"]]
    for run in runs:
        found, lines = evaluate_run(run, settings, ground_truths, device, out_dir)
        rows += found
        checks += lines
    text = report(rows, settings, [run.name for run in runs], checks)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    return text
