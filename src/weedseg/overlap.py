"""Overlapping frames: does the classifier do better on several real images of a plant than on one?

Test-time augmentation (weedseg.tta) shows the classifier one image in several ways. A second frame
is a new image: another exposure, another angle, and a plant cut by one frame's border may be whole
in the next. Nothing about the robot is assumed except that its images overlap, so frames are placed
from the images alone:

- weedseg.frames places a frame on a neighbour in its session (SIFT and RANSAC). A pair must have
  moved, which drops a robot standing still and marks on the rig matched in every frame, neither of
  which is a new view;
- `link_frames`: boxes are moved into the neighbour and joined by IoU, inside the part both frames
  see. Linked boxes are one plant;
- `combined`: every view of a plant gets the same probabilities. `mean` is what the views' average
  says, `product` treats them as independent evidence, which sharpens, and `whole first` trusts
  views not cut by the frame's border.

A plant seen once keeps its probabilities exactly, so whatever moves comes from the plants seen
more than once. All frames of a plant are used, later ones too: the decision waits until it has
left the camera. The classifier isn't run: probabilities come from the two-stage evaluation
(detections) and
from scripts/evaluate_tta.py's kept views (labelled boxes, and the same plants' augmented views as a
control), so this needs a CPU and the images.

A link between two labelled boxes is also a check of the labels: one plant should have one label.

scripts/evaluate_overlap.py runs it. See docs/twostage.md (Overlapping frames).
"""

from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

from weedseg.classifier.crops import read_index
from weedseg.compare import run_dir, twostage_eval
from weedseg.config import load_config
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.frames import frame_groups, frame_shift, registrations, usable
from weedseg.maskrcnn.data import load_coco
from weedseg.maskrcnn.engine import CLASS_SCORE_MIN
from weedseg.maskrcnn.run import split_file
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
from weedseg.tta import PLAIN, as_saved, collect_views, count_spread, crop_evaluation, partial_fit

METHODS = ("mean", "product", "whole first")
ONE_FRAME = "augmented views, one frame"
ALL_FRAMES = "augmented views, all frames"
PLANTS_FROM = {"crops": "Classifier alone, on the labelled boxes",
               "detections": "Both stages, on the plants found (datasets with masks only)"}
SEEN = {"species_right_seen": "species right, seen twice", "decided": "decided by classifier",
        "right_decided": "right among decided"}
COUNTS = {"wrong_to_right": "wrong to right", "right_to_wrong": "right to wrong",
          "fixes_over_hand_off": "fixes over hand-off"}


def project(box: list[float], matrix: list[list[float]]) -> tuple[float, float, float, float]:
    """The corners (x0, y0, x1, y1) around a box [x, y, w, h] moved by `matrix`."""
    x, y, w, h = box
    corners = np.array([[x, y, 1.0], [x + w, y, 1.0], [x + w, y + h, 1.0], [x, y + h, 1.0]]).T
    moved = np.asarray(matrix) @ corners
    return moved[0].min(), moved[1].min(), moved[0].max(), moved[1].max()


def corners_iou(a: tuple, b: tuple) -> float:
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    both = width * height
    return float(both / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both))


def link_frames(boxes_of: dict[int, list[tuple[int, list[float]]]],
                sizes: dict[int, tuple[int, int]], pairs: list[dict[str, Any]],
                min_iou: float) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    """Which boxes are one plant. `boxes_of`: per image id, (row, box [x, y, w, h]). In each pair of
    frames, a box is moved into the other frame, cut to what that frame sees, and joined one to one
    with its best match. Returns the plants seen more than once, as rows, and the links."""
    parent = {}

    def root(row: int) -> int:
        parent.setdefault(row, row)
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = parent[row]
        return row

    links = []
    for pair in pairs:
        width, height = sizes[pair["b"]]
        options = []
        for row, box in boxes_of.get(pair["a"], []):
            x0, y0, x1, y1 = project(box, pair["matrix"])
            seen = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
            if seen[2] <= seen[0] or seen[3] <= seen[1]:
                continue
            for other, (x, y, w, h) in boxes_of.get(pair["b"], []):
                value = corners_iou(seen, (x, y, x + w, y + h))
                if value >= min_iou:
                    options.append((value, row, other))
        used = set()
        for value, row, other in sorted(options, reverse=True):
            if row not in used and other not in used:
                used |= {row, other}
                parent[root(row)] = root(other)
                links.append((row, other, value))
    tracks = defaultdict(list)
    for row in sorted(parent):
        tracks[root(row)].append(row)
    return list(tracks.values()), links


def cut_off(box: list[float], size: tuple[int, int], border: float) -> bool:
    """Whether a box touches the border of its frame, so the plant may be cut."""
    x, y, w, h = box
    return x <= border or y <= border or x + w >= size[0] - border or y + h >= size[1] - border


def combine_views(probs: np.ndarray, method: str, whole: np.ndarray) -> np.ndarray:
    """One plant's probabilities from its views [views, classes]; `whole`: views not cut off."""
    if method == "mean":
        return probs.mean(0)
    if method == "whole first":
        return probs[whole].mean(0) if whole.any() else probs.mean(0)
    if method == "product":  # floored: the pipeline drops classes under CLASS_SCORE_MIN
        logged = np.log(np.maximum(probs, CLASS_SCORE_MIN)).sum(0)
        found = np.exp(logged - logged.max())
        return found / found.sum()
    raise ValueError(f"method must be one of {METHODS}, not {method!r}")


def combined(plain: np.ndarray, tracks: list[list[int]], method: str,
             whole: np.ndarray) -> np.ndarray:
    """`plain` with each plant's views combined, treated as the pipeline treats its own
    probabilities. Plants seen once stay exactly as they are."""
    found = plain.copy()
    for track in tracks:
        found[track] = combine_views(plain[track], method, whole[track])
    rows = [row for track in tracks for row in track]
    found[rows] = as_saved(found[rows])
    return found


class PlantSet(NamedTuple):
    """One set of plants to score, row by row."""

    kind: str  # a key of PLANTS_FROM
    evaluation: Evaluation
    plain: np.ndarray  # [rows, classes], as the pipeline keeps them
    augmented: np.ndarray  # the mean of each row's augmented views, likewise
    boxes: list[list[float]]
    image_ids: list[int]
    groups: list[str]  # the session of each row
    rows: list[dict[str, Any]] | None = None  # crop index rows, for the partial-label fit


def plant_links(plants: PlantSet, pairs: dict[str, list[dict[str, Any]]],
                sizes: dict[int, tuple[int, int]],
                min_iou: float) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    """`link_frames` for each dataset with registered frames."""
    tracks, links = [], []
    for dataset, found in pairs.items():
        boxes_of = defaultdict(list)
        for row, session in enumerate(plants.evaluation.sessions):
            if session[0] == dataset:
                boxes_of[plants.image_ids[row]].append((row, plants.boxes[row]))
        here = link_frames(boxes_of, sizes, found, min_iou)
        tracks += here[0]
        links += here[1]
    return tracks, links


def variants(plants: PlantSet, tracks: list[list[int]], whole: np.ndarray) -> dict[str, np.ndarray]:
    """Each way of treating the plants seen more than once. The augmented views are the control:
    on the same plants, what the views of one frame give, without and with the other frames."""
    linked = [row for track in tracks for row in track]
    one_frame, all_frames = plants.plain.copy(), plants.plain.copy()
    one_frame[linked] = plants.augmented[linked]
    all_frames[linked] = combined(plants.augmented, tracks, "mean", whole)[linked]
    return {PLAIN: plants.plain,
            **{method: combined(plants.plain, tracks, method, whole) for method in METHODS},
            ONE_FRAME: one_frame, ALL_FRAMES: all_frames}


def _is_named(evaluation: Evaluation, true: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Plants `species_right` counts: other_plant is neither their label nor their top class."""
    other = np.array([name == OTHER_CATEGORY for name in evaluation.classes])
    return ~other[true] & ~other[top]


def evaluate_set(plants: PlantSet, tracks: list[list[int]], whole: np.ndarray,
                 settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Per variant and dataset: `scores`; the plants seen more than once; what the variant did to
    the top classes against plain; and who decides under the export's hand-off to the zero-shot
    model, since a fix under it never reaches the output."""
    evaluation = plants.evaluation
    labelled = np.flatnonzero(evaluation.true >= 0)
    true = evaluation.true[labelled]
    datasets = np.array([evaluation.sessions[row][0] for row in labelled])
    seen = np.zeros(len(plants.plain), bool)
    seen[[row for track in tracks for row in track]] = True
    seen = seen[labelled]
    before = plants.plain[labelled].argmax(1)
    keep = {ALL, *settings["datasets"]}
    found = []
    for name, probs in variants(plants, tracks, whole).items():
        top, best = probs[labelled].argmax(1), probs[labelled].max(1)
        fits = partial_fit(plants.rows, probs, evaluation.classes) if plants.rows else {}
        for group, values in scores(evaluation, probs, settings).items():
            if group not in keep:
                continue
            mask = np.ones(len(labelled), bool) if group == ALL else datasets == group
            was_right, is_right = before == true, top == true
            fixed, decided = ~was_right & is_right, best >= settings["hand_off_below"]
            named = _is_named(evaluation, true, top) & mask & seen
            found.append({
                "plants_from": plants.kind, "dataset": group, "variant": name, **values,
                "partial_fit": fits.get(group), "seen_twice": int((mask & seen).sum()),
                "species_right_seen": float(is_right[named].mean()) if named.any() else None,
                "wrong_to_right": int((fixed & mask).sum()),
                "right_to_wrong": int((was_right & ~is_right & mask).sum()),
                "fixes_over_hand_off": int((fixed & decided & mask).sum()),
                "decided": float(decided[mask].mean()) if mask.any() else None,
                "right_decided": (float(is_right[mask & decided].mean())
                                  if (mask & decided).any() else None)})
    return found


def session_counts(evaluation: Evaluation, probs: np.ndarray, groups: list[str],
                   rows: np.ndarray) -> np.ndarray:
    """Per session, in sorted order: the labelled `rows` with the right species, and those counted
    (as `species_right`). [sessions, 2]"""
    plants = rows[evaluation.true[rows] >= 0]
    true, top = evaluation.true[plants], probs[plants].argmax(1)
    named = _is_named(evaluation, true, top)
    counts = defaultdict(lambda: [0, 0])
    for row, right, counted in zip(plants, top == true, named):
        counts[groups[row]][0] += int(right and counted)
        counts[groups[row]][1] += int(counted)
    return np.array([counts[group] for group in sorted(counts)], float).reshape(-1, 2)


def gain_interval(before: np.ndarray, after: np.ndarray, draws: int,
                  seed: int = 0) -> tuple[float, float, float]:
    """The gain in the right share and its 95% interval, resampling sessions, the mean of the runs.
    `before`, `after`: [runs, sessions, 2], the same sessions, as `session_counts`."""
    sessions = before.shape[1]
    weights = np.random.default_rng(seed).multinomial(sessions, np.full(sessions, 1 / sessions),
                                                      size=draws).astype(float)

    def share(counts: np.ndarray) -> np.ndarray:
        right, total = weights @ counts[..., 0].T, weights @ counts[..., 1].T
        return np.divide(right, total, out=np.zeros_like(right), where=total > 0)

    gains = (share(after) - share(before)).mean(1)
    point = (after[..., 0].sum(1) / after[..., 1].sum(1)
             - before[..., 0].sum(1) / before[..., 1].sum(1)).mean()
    low, high = np.percentile(gains, [2.5, 97.5])
    return float(point), float(low), float(high)


def interval_counts(plants: PlantSet, tracks: list[list[int]], whole: np.ndarray,
                    settings: dict[str, Any]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """`session_counts` per dataset, variant and case: all a dataset's plants, or those seen more
    than once. All `intervals` needs of a run, so the run's plants needn't stay in memory."""
    seen = np.zeros(len(plants.plain), bool)
    seen[[row for track in tracks for row in track]] = True
    probs_of = variants(plants, tracks, whole)
    found = {}
    for dataset in settings["datasets"]:
        here = np.array([session[0] == dataset for session in plants.evaluation.sessions])
        cases = {"all": np.flatnonzero(here), "seen": np.flatnonzero(here & seen)}
        found[dataset] = {
            name: {case: session_counts(plants.evaluation, probs, plants.groups, rows)
                   for case, rows in cases.items()}
            for name, probs in probs_of.items()}
    return found


def intervals(kind: str, per_run: list[dict[str, dict[str, dict[str, np.ndarray]]]],
              settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Per variant and dataset of one set of plants (`interval_counts` of each run): the gain in
    species right against plain, over all its plants and over those seen more than once. Only with
    `min_sessions` sessions: fewer can't show what another session would do."""
    found = []
    for dataset in settings["datasets"]:
        for name in per_run[0][dataset]:
            if name == PLAIN:
                continue
            row = {"plants_from": kind, "dataset": dataset, "variant": name}
            for case in ("all", "seen"):
                before = [run[dataset][PLAIN][case] for run in per_run]
                after = [run[dataset][name][case] for run in per_run]
                if len({len(counts) for counts in (*before, *after)}) > 1:
                    continue  # a session only some runs count: no paired resampling
                row[f"sessions_{case}"] = len(before[0])
                if len(before[0]) >= settings["min_sessions"]:
                    point, low, high = gain_interval(np.array(before), np.array(after),
                                                     settings["interval_draws"])
                    row |= {f"gain_{case}": point, f"gain_{case}_low": low,
                            f"gain_{case}_high": high}
            found.append(row)
    return found


def link_summary(plants: PlantSet, tracks: list[list[int]], links: list[tuple[int, int, float]],
                 settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Per dataset: how many labelled plants are seen more than once, and whether a link holds one
    plant: the same label at both ends, and the same top class."""
    evaluation = plants.evaluation
    dataset_of = [session[0] for session in evaluation.sessions]
    top = plants.plain.argmax(1)
    views = np.ones(len(plants.plain), int)
    for track in tracks:
        views[track] = len(track)
    found = []
    for dataset in settings["datasets"]:
        labelled = [row for row, name in enumerate(dataset_of)
                    if name == dataset and evaluation.true[row] >= 0]
        if not labelled:
            continue
        here = [link for link in links if dataset_of[link[0]] == dataset]
        both = [(a, b) for a, b, _ in here if evaluation.true[a] >= 0 and evaluation.true[b] >= 0]
        counted = Counter(views[labelled].tolist())
        found.append({
            "plants_from": plants.kind, "dataset": dataset, "plants": len(labelled),
            "seen_twice": len(labelled) - counted[1],
            "seen_twice_share": 1 - counted[1] / len(labelled),
            "seen_in_2": counted[2],
            "seen_in_3_or_more": sum(n for k, n in counted.items() if k > 2),
            "links": len(here), "both_labelled": len(both),
            "same_label": (sum(evaluation.true[a] == evaluation.true[b] for a, b in both)
                           / len(both)) if both else None,
            "same_top": (sum(top[a] == top[b] for a, b, _ in here) / len(here)) if here else None})
    return found


def label_conflicts(plants: PlantSet, links: list[tuple[int, int, float]], coco_images: dict,
                    annotation_ids: list[int] | None, run: str) -> list[dict[str, Any]]:
    """The links whose two ends have different labels: one plant named twice, or a wrong link."""
    evaluation = plants.evaluation
    top, best = plants.plain.argmax(1), plants.plain.max(1)
    found = []
    for a, b, value in links:
        labels = evaluation.true[a], evaluation.true[b]
        if min(labels) < 0 or labels[0] == labels[1]:
            continue
        row = {"run": run, "dataset": evaluation.sessions[a][0], "iou": round(value, 3)}
        for side, end in (("a", a), ("b", b)):
            row |= {f"label_{side}": evaluation.classes[evaluation.true[end]],
                    f"called_{side}": evaluation.classes[top[end]],
                    f"probability_{side}": round(float(best[end]), 4),
                    f"file_{side}": coco_images[plants.image_ids[end]]["file_name"],
                    f"box_{side}": " ".join(f"{v:.0f}" for v in plants.boxes[end])}
            if annotation_ids is not None:
                row[f"annotation_{side}"] = annotation_ids[end]
        found.append(row)
    return found


def frame_summary(dataset: str, groups: dict[str, list[dict[str, Any]]],
                  pairs: list[dict[str, Any]], kept: list[dict[str, Any]],
                  sizes: dict[int, tuple[int, int]], settings: dict[str, Any]) -> dict[str, Any]:
    """How much of a dataset's frames overlap."""
    shifts = [frame_shift(pair["matrix"], sizes[pair["a"]]) for pair in kept]
    next_ones = {(a["id"], b["id"]) for items in groups.values() for a, b in pairwise(items)}
    return {
        "dataset": dataset, "sessions": len(groups),
        "frames": sum(len(items) for items in groups.values()),
        "neighbours": len(next_ones),
        "neighbours_overlapping": sum((pair["a"], pair["b"]) in next_ones for pair in kept),
        "pairs_tried": sum(max(0, len(items) - gap) for items in groups.values()
                           for gap in settings["gaps"]),
        "pairs_matched": len(pairs), "pairs_usable": len(kept),
        "sessions_overlapping": len({info["group"] for items in groups.values() for info in items
                                     if info["id"] in {pair["a"] for pair in kept}}),
        "median_shift": float(np.median(shifts)) if shifts else None}


def evaluate_run(run: Path, settings: dict[str, Any], tta: dict[str, Any],
                 pairs: dict[str, list[dict[str, Any]]], image: dict[int, dict[str, Any]],
                 box_of: dict[int, list[float]],
                 ground_truths: dict[Path, dict]) -> dict[str, Any]:
    """The plants of a classifier run, both sets. `image`: the split's images by id; `box_of`: its
    annotations' boxes by id."""
    split, checkpoint = settings["split"], settings["checkpoint"]
    folder = twostage_eval(resolve(settings["twostage_dir"]), run, split, checkpoint)
    if folder is None:
        raise SystemExit(f"No two-stage evaluation of {run.name}, {split} {checkpoint}; "
                         "run scripts/predict_twostage.py")
    evaluation = load_evaluation(folder, ground_truths)
    base = resolve(tta["output_dir"]) / run.name / f"{split}_{checkpoint}"
    if not (base / "crops" / "plants.csv").exists():
        raise SystemExit(f"No kept views in {base}; run scripts/evaluate_tta.py first")

    def missing(_view: dict[str, Any]) -> None:
        raise SystemExit(f"A view is missing in {base}; run scripts/evaluate_tta.py first")

    def augmented(views: dict[str, np.ndarray]) -> np.ndarray:
        return as_saved(np.mean(list(views.values()), 0))

    rows = read_index(base / "crops" / "plants.csv")
    sizes = {image_id: (info["width"], info["height"]) for image_id, info in image.items()}
    crop_views = collect_views(base / "crops", np.array([row["annotation_id"] for row in rows]),
                               tta["views"], missing, evaluation.classes)
    detection_ids = np.array([[image_id, *box] for image_id, box
                              in zip(evaluation.image_ids, evaluation.boxes)])
    detection_views = collect_views(base / "detections", detection_ids, tta["views"], missing,
                                    evaluation.classes)
    sets = [
        PlantSet("crops", crop_evaluation(evaluation, rows), as_saved(crop_views[PLAIN]),
                 augmented(crop_views), [box_of[row["annotation_id"]] for row in rows],
                 [row["image_id"] for row in rows],
                 [image[row["image_id"]]["group"] for row in rows], rows),
        PlantSet("detections", evaluation, evaluation.probs, augmented(detection_views),
                 evaluation.boxes, evaluation.image_ids,
                 [session[1] for session in evaluation.sessions])]
    scored, links_found, conflicts, counts = [], [], [], {}
    for plants in sets:
        tracks, links = plant_links(plants, pairs, sizes, settings["min_iou"])
        whole = np.array([not cut_off(box, sizes[image_id], settings["border_px"])
                          for box, image_id in zip(plants.boxes, plants.image_ids)])
        scored += [{"run": run.name, **row}
                   for row in evaluate_set(plants, tracks, whole, settings)]
        links_found += [{"run": run.name, **row}
                        for row in link_summary(plants, tracks, links, settings)]
        counts[plants.kind] = interval_counts(plants, tracks, whole, settings)
        if plants.kind == "crops":
            conflicts += label_conflicts(plants, links, image,
                                         [row["annotation_id"] for row in rows], run.name)
    return {"scored": scored, "links": links_found, "conflicts": conflicts, "counts": counts}


def conflict_table(conflicts: list[dict[str, Any]]) -> list[str]:
    """Which labels disagree across a link, and what the classifier calls those views."""
    if not conflicts:
        return ["No link joins two different labels.", ""]
    runs = sorted({row["run"] for row in conflicts})
    groups = defaultdict(lambda: defaultdict(list))
    for row in conflicts:
        pair = tuple(sorted((row["label_a"], row["label_b"])))
        groups[row["dataset"], *pair][row["run"]].append(row)
    lines = ["| dataset | labels | links | views called the first | called the second | other |",
             "|---|---|---|---|---|---|"]
    order = sorted(groups, key=lambda key: -sum(len(rows) for rows in groups[key].values()))
    for key in order:
        dataset, first, second = key
        shares = {first: [], second: [], "other": []}
        for run in runs:
            calls = [row[f"called_{side}"] for row in groups[key][run] for side in "ab"]
            for name in (first, second):
                shares[name].append(calls.count(name) / max(len(calls), 1))
            shares["other"].append(sum(c not in (first, second) for c in calls)
                                   / max(len(calls), 1))
        count = len(groups[key][runs[0]])
        lines.append(f"| {dataset} | {first}, {second} | {count} | {spread(shares[first], 0)} | "
                     f"{spread(shares[second], 0)} | {spread(shares['other'], 0)} |")
    return [*lines, ""]


def report(results: dict[str, list], frames: list[dict[str, Any]], settings: dict[str, Any],
           runs: list[str]) -> str:
    """The Markdown report."""
    targets = settings["crops_removed"]
    shares = {**SHARES, **{f"harmful_at_{t}": f"harmful at <={t:.0%} crops" for t in targets},
              "partial_fit": "partial-label fit", **SEEN}
    intro = ("Percentages of the labelled plants, each view of a plant counted; a range is the "
             "spread across runs, and the plain row's spread is the noise to compare with. Only "
             "plants seen more than once change, so a dataset's rows show what its overlap gives, "
             "and 'seen twice' what plants with a second view gain. Compare decisions at the same "
             "crops removed. Fixes over hand-off: fixes at or over the export's hand-off to the "
             "zero-shot model; other fixes never reach its output. See docs/twostage.md "
             "(Overlapping frames).")
    lines = [f"Runs: {', '.join(runs)}.", "", intro, "", "# Frames", "",
             ("| dataset | sessions | frames | next frames overlapping | sessions overlapping | "
             "pairs matched, usable | median move |"), "|---|---|---|---|---|---|---|"]
    for row in frames:
        shift = "-" if row["median_shift"] is None else f"{row['median_shift']:.0%}"
        lines.append(f"| {row['dataset']} | {row['sessions']} | {row['frames']} | "
                     f"{row['neighbours_overlapping']} of {row['neighbours']} | "
                     f"{row['sessions_overlapping']} | {row['pairs_matched']}, "
                     f"{row['pairs_usable']} | {shift} |")
    lines += ["", "# Links", "",
              ("| plants from | dataset | plants | seen twice | in 2 frames | in 3 or more | "
               "links | both labelled | same label | same top class |"),
              "|---" * 10 + "|"]
    links = results["links"]
    for kind in PLANTS_FROM:
        for dataset in settings["datasets"]:
            here = [row for row in links
                    if row["plants_from"] == kind and row["dataset"] == dataset]
            if here:
                row = here[0]
                lines.append(
                    f"| {kind} | {dataset} | {row['plants']} | "
                    f"{row['seen_twice']} ({row['seen_twice_share']:.1%}) | {row['seen_in_2']} | "
                    f"{row['seen_in_3_or_more']} | {row['links']} | {row['both_labelled']} | "
                    f"{spread([r['same_label'] for r in here], 1)} | "
                    f"{spread([r['same_top'] for r in here], 1)} |")
    lines += ["", "# Labels across a link", "",
              "One plant should have one label. The full list is in <report>_label_conflicts.csv.",
              "", *conflict_table(results["conflicts"])]
    scored = results["scored"]
    for kind, title in PLANTS_FROM.items():
        chosen = [row for row in scored if row["plants_from"] == kind]
        columns = {key: name for key, name in shares.items()
                   if any(row.get(key) is not None for row in chosen)}
        lines += [f"# {title}", ""]
        for dataset in sorted({row["dataset"] for row in chosen},
                              key=lambda name: (name != ALL, name)):
            here = [row for row in chosen if row["dataset"] == dataset]
            lines += [(f"## {dataset}: {here[0]['plants']} plants, "
                      f"{here[0]['seen_twice']} seen more than once"), "",
                      "| variant | " + " | ".join([*columns.values(), *COUNTS.values()]) + " |",
                      "|---" * (1 + len(columns) + len(COUNTS)) + "|"]
            for variant in dict.fromkeys(row["variant"] for row in here):
                cases = [row for row in here if row["variant"] == variant]
                cells = [spread([row[key] for row in cases], 2) for key in columns]
                cells += [count_spread([row[key] for row in cases]) for key in COUNTS]
                lines.append(f"| {variant} | " + " | ".join(cells) + " |")
            lines.append("")
    lines += ["# Gain in species right against plain", "",
              (f"Points, the mean of the runs, with a 95% interval resampling sessions (only with "
              f"at least {settings['min_sessions']}). All: over the dataset's plants. Seen twice: "
              "over those with a second view."), "",
              "| plants from | dataset | variant | all | seen twice |", "|---|---|---|---|---|"]
    for row in results["intervals"]:
        cells = [f"{row[f'gain_{key}'] * 100:+.2f} ({row[f'gain_{key}_low'] * 100:+.2f} to "
                 f"{row[f'gain_{key}_high'] * 100:+.2f})" if f"gain_{key}" in row else "-"
                 for key in ("all", "seen")]
        lines.append(f"| {row['plants_from']} | {row['dataset']} | {row['variant']} | "
                     f"{cells[0]} | {cells[1]} |")
    return "\n".join([*lines, ""])


def write_report(settings: dict[str, Any], stem: Path) -> str:
    """Register what isn't kept yet and score every variant. Writes <stem>.md, <stem>.csv (every
    number per run), <stem>_links.csv, <stem>_intervals.csv and <stem>_label_conflicts.csv; returns
    the Markdown."""
    tta = load_config(settings["tta_config"])
    runs = [run_dir(entry) for entry in settings["runs"]]
    first = yaml.safe_load((runs[0] / "config.yaml").read_text(encoding="utf-8"))
    coco = load_coco(split_file(first, settings["split"]))
    images = coco["images"]
    image_dir = resolve(coco["info"]["image_dir"])
    box_of = {item["id"]: item["bbox"] for item in coco["annotations"]}
    image = {info["id"]: info for info in images}
    sizes = {info["id"]: (info["width"], info["height"]) for info in images}
    out_dir = resolve(settings["output_dir"])
    pairs, frames = {}, []
    for dataset in settings["datasets"]:
        groups = frame_groups(images, dataset)
        found = registrations(groups, image_dir, settings,
                              out_dir / "registrations" / f"{settings['split']}_{dataset}.json",
                              settings["num_workers"])
        pairs[dataset] = [pair for pair in found if usable(pair, sizes[pair["a"]], settings)]
        frames.append(frame_summary(dataset, groups, found, pairs[dataset], sizes, settings))
    results = {"scored": [], "links": [], "conflicts": [], "intervals": []}
    counts = defaultdict(list)
    ground_truths = {}
    for run in runs:
        found = evaluate_run(run, settings, tta, pairs, image, box_of, ground_truths)
        for key in ("scored", "links", "conflicts"):
            results[key] += found[key]
        for kind, per_dataset in found["counts"].items():
            counts[kind].append(per_dataset)
    for kind in PLANTS_FROM:
        results["intervals"] += intervals(kind, counts[kind], settings)
    text = report(results, frames, settings, [run.name for run in runs])
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), results["scored"])
    write_csv(stem.with_name(f"{stem.name}_links.csv"), results["links"])
    write_csv(stem.with_name(f"{stem.name}_intervals.csv"), results["intervals"])
    write_csv(stem.with_name(f"{stem.name}_label_conflicts.csv"), results["conflicts"])
    return text
