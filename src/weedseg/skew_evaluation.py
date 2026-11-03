"""What a skew by a crop list and a weed list (weedseg.skew) does to a two-stage evaluation, made
again from its saved predictions: no GPU and no retraining.

The skew step doesn't care where its lists come from, but an evaluation needs some. These
stand-ins give each session (an image's dataset and `group`) its lists:

- `labelled`: the species labelled in the session. The best case: nothing is missing, and a
  session labels few species, so lists are shorter than a real expert's.
- `first_pass`: from the model's raw predictions, the most frequent crop class and the weeds
  that make up at least `weed_share` of the plants called weeds. Sessions with fewer than
  `min_plants` detections get no list, since one misnamed plant would count too much.
- `first_pass_paired`: the same, but sessions under `min_plants` are first joined two by two,
  those with the same most frequent crop class, the largest first. For testing only: many
  CropAndWeed sessions are short.

Each list can be used alone or with the other (`lists`). Reported per dataset: the species and
crop or non-crop calls on the plants found, decisions at the rule file's defer_below and at the
same shares of crops removed, and what a list costs a weed species it misses. See
docs/twostage.md (Species lists).
"""

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from weedseg.compare import twostage_eval
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import Rules, harmful_removed_at, load_rules, removed_up_to, role_of
from weedseg.eppo import load_taxa
from weedseg.evaluation import ignore_category_ids, match_detections
from weedseg.maskrcnn.data import load_masked_coco
from weedseg.paths import resolve
from weedseg.skew import skew_probabilities

Session = tuple[str, str]  # dataset, group
SIDES = {"both": (True, True), "crop": (True, False), "weed": (False, True)}
ALL = "all"
# Columns of the report's tables: the plants found, then decisions.
SHARES = {"species_right": "species right", "class_right": "class right", "balanced": "balanced",
          "crop_right": "crop or not right", "weeds_called_crop": "weeds called crop",
          "under_hand_off": "under hand-off", "crops_removed": "crops removed",
          "harmful_removed": "harmful removed"}
MISSED = {"right_unskewed": "right, no skew", "right": "right, skewed",
          "crop_unskewed": "called crop, no skew", "crop": "called crop, skewed"}


@dataclass(frozen=True)
class Lists:
    """The lists of a session, as class names; `partner` is the session it was joined with."""

    crop: frozenset[str] = frozenset()
    weed: frozenset[str] = frozenset()
    partner: str = ""


@dataclass
class Evaluation:
    """The detections of a two-stage evaluation that reach its score threshold."""

    classes: list[str]
    crops: set[str]  # classes tagged as crops
    ids: list[int]  # category id of each class
    roles: dict[str, str]
    rules: Rules
    probs: np.ndarray  # [detections, classes], raw, each row scaled to sum to 1
    plant_scores: np.ndarray
    sessions: list[Session]  # of each detection
    true: np.ndarray  # class index of the labelled plant matched to each detection, or -1
    labelled: dict[Session, set[str]]  # classes labelled per session, found or not

    @cached_property
    def rows(self) -> dict[Session, np.ndarray]:
        """The detections of each session."""
        rows = defaultdict(list)
        for row, session in enumerate(self.sessions):
            rows[session].append(row)
        return {session: np.array(found) for session, found in rows.items()}

    @cached_property
    def is_crop(self) -> np.ndarray:
        return np.array([name in self.crops for name in self.classes])

    @cached_property
    def is_weed(self) -> np.ndarray:
        return np.array([name != OTHER_CATEGORY for name in self.classes]) & ~self.is_crop


def load_evaluation(folder: Path, ground_truths: dict[Path, dict]) -> Evaluation:
    """An evaluation folder of predict_twostage.py, with the ground truth, matching and rule file
    of its evaluation.json; the rule file as it is now. `ground_truths` keeps each ground truth
    loaded, by path."""
    evaluation = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))
    settings = evaluation["settings"]
    if settings.get("max_images") is not None:
        raise ValueError(f"{folder} was evaluated on {settings['max_images']} images only")
    path = resolve(evaluation["split_file"])
    if path not in ground_truths:
        ground_truths[path] = load_masked_coco(path)
    gt = ground_truths[path]
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    rules = load_rules(resolve(settings["rules"]), taxa)
    ignore_ids = ignore_category_ids(gt)
    categories = [c for c in gt["categories"] if c["id"] not in ignore_ids]
    classes = [c["name"] for c in categories]
    column = {c["id"]: k for k, c in enumerate(categories)}
    session_of = {i["id"]: (i.get("dataset") or "(none)", i.get("group") or "(none)")
                  for i in gt["images"]}
    results = [r for r in json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
               if r["score"] >= settings["score_threshold"]]
    matched = {id(m.detection): column[m.plant["category_id"]]
               for m in match_detections(gt, results, None, settings["score_threshold"],
                                         settings["iou_threshold"], settings["match_by"])
               if m.plant and m.detection}
    probs = np.zeros((len(results), len(classes)))
    for row, result in enumerate(results):
        for category_id, probability in result["class_scores"].items():
            if int(category_id) in column:
                probs[row, column[int(category_id)]] = probability
    total = probs.sum(1, keepdims=True)
    labelled = defaultdict(set)
    for annotation in gt["annotations"]:
        if annotation["category_id"] in column and not annotation.get("iscrowd"):
            labelled[session_of[annotation["image_id"]]].add(
                classes[column[annotation["category_id"]]])
    return Evaluation(
        classes, {c["name"] for c in categories if c.get("supercategory") == "crop"},
        [c["id"] for c in categories], {name: role_of(name, rules, taxa) for name in classes},
        rules, np.divide(probs, total, out=probs, where=total > 0),
        np.array([r.get("plant_score", r["score"]) for r in results]),
        [session_of[r["image_id"]] for r in results],
        np.array([matched.get(id(r), -1) for r in results], dtype=np.int64), dict(labelled))


def labelled_lists(evaluation: Evaluation) -> dict[Session, Lists]:
    """The species labelled in each session."""
    return {session: Lists(frozenset(names & evaluation.crops),
                           frozenset(names - evaluation.crops - {OTHER_CATEGORY}))
            for session, names in evaluation.labelled.items()}


def _top_crop(evaluation: Evaluation, counts: np.ndarray) -> str | None:
    crops = counts * evaluation.is_crop
    return evaluation.classes[int(crops.argmax())] if crops.any() else None


def _first_pass_list(evaluation: Evaluation, counts: np.ndarray, weed_share: float,
                     partner: str = "") -> Lists:
    weeds = counts * evaluation.is_weed
    crop = _top_crop(evaluation, counts)
    return Lists(frozenset({crop} if crop else ()),
                 frozenset(evaluation.classes[k] for k in np.flatnonzero(weeds)
                           if weeds[k] >= weed_share * weeds.sum()), partner)


def first_pass_lists(evaluation: Evaluation, weed_share: float, min_plants: int,
                     pair_small: bool = False) -> dict[Session, Lists]:
    """Lists from the model's raw predictions (see module doc); no entry for a session with too
    few plants, even when joined."""
    top = evaluation.probs.argmax(1)
    counts = {session: np.bincount(top[rows], minlength=len(evaluation.classes))
              for session, rows in evaluation.rows.items()}
    groups = [[session] for session, found in counts.items() if found.sum() >= min_plants]
    if pair_small:
        small = defaultdict(list)
        for session, found in counts.items():
            if found.sum() < min_plants:
                small[session[0], _top_crop(evaluation, found)].append(session)
        for members in small.values():
            members.sort(key=lambda session: (-counts[session].sum(), session))
            groups += [members[i:i + 2] for i in range(0, len(members), 2)]
    lists = {}
    for group in groups:
        joined = sum(counts[session] for session in group)
        if joined.sum() >= min_plants:
            for session in group:
                partner = next((other[1] for other in group if other != session), "")
                lists[session] = _first_pass_list(evaluation, joined, weed_share, partner)
    return lists


def skewed(evaluation: Evaluation, lists: dict[Session, Lists], side: str,
           multiplier: float) -> np.ndarray:
    """All detections' probabilities, skewed by their session's lists; `side` is a key of SIDES."""
    use_crop, use_weed = SIDES[side]
    probs = evaluation.probs.copy()
    for session, found in lists.items():
        if session in evaluation.rows:
            rows = evaluation.rows[session]
            probs[rows] = skew_probabilities(
                evaluation.probs[rows], evaluation.classes, evaluation.crops,
                found.crop if use_crop else (), found.weed if use_weed else (),
                use_lists=True, multiplier=multiplier)
    return probs


def _removed_up_to(evaluation: Evaluation, probs: np.ndarray, row: int) -> float | None:
    names = dict(zip(evaluation.ids, evaluation.classes))
    plant = float(evaluation.plant_scores[row])
    detection = {"plant_score": plant, "score": plant,
                 "category_id": evaluation.ids[int(probs[row].argmax())],
                 "class_scores": {str(evaluation.ids[k]): float(probs[row, k])
                                  for k in np.flatnonzero(probs[row])}}
    return removed_up_to(detection, names, evaluation.roles, evaluation.rules.plant_defer_below)


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def scores(evaluation: Evaluation, probs: np.ndarray, settings: dict[str, Any]) -> dict[str, dict]:
    """Per dataset and ALL, on the labelled plants found: the shares of SHARES, and for each of
    settings["crops_removed"] the harmful removed at that share of crops removed, with the
    defer_below that gives it.

    `species_right` is the two-stage evaluation's (by_dataset.csv): plants with other_plant as
    their label or their top class are left out, so without a skew the two agree. `class_right`
    counts them, so a named plant moved to other_plant is wrong there."""
    plants = np.flatnonzero(evaluation.true >= 0)
    true, top = evaluation.true[plants], probs[plants].argmax(1)
    roles = np.array([evaluation.roles[evaluation.classes[k]] for k in true])
    datasets = np.array([evaluation.sessions[row][0] for row in plants])
    removable = [_removed_up_to(evaluation, probs, row) for row in plants]
    defer_below = evaluation.rules.defer_below or 0
    removed = np.array([value is not None and value >= defer_below for value in removable])
    best = probs[plants].max(1)
    is_other = np.array([name == OTHER_CATEGORY for name in evaluation.classes])
    named = ~is_other[true] & ~is_other[top]
    found = {}
    for group in (ALL, *sorted(set(datasets.tolist()))):
        mask = np.ones(len(plants), bool) if group == ALL else datasets == group
        per_class = [(top[mask & (true == k)] == k).mean() for k in np.unique(true[mask])
                     if (mask & (true == k)).sum() >= settings["min_class_plants"]]
        weeds = mask & evaluation.is_weed[true]
        crops, harmful = mask & (roles == "crop"), mask & (roles == "harmful")
        row = {"plants": int(mask.sum()),
               "species_right": _mean(top[mask & named] == true[mask & named]),
               "class_right": _mean(top[mask] == true[mask]),
               "balanced": float(np.mean(per_class)) if per_class else None,
               "crop_right": _mean(evaluation.is_crop[top[mask]] == evaluation.is_crop[true[mask]]),
               "weeds_called_crop": _mean(evaluation.is_crop[top[weeds]]),
               "under_hand_off": _mean(best[mask] < settings["hand_off_below"]),
               "crops_removed": _mean(removed[crops]), "harmful_removed": _mean(removed[harmful])}
        for target in settings["crops_removed"]:
            row[f"harmful_at_{target}"], row[f"defer_below_at_{target}"] = harmful_removed_at(
                [removable[i] for i in np.flatnonzero(crops)],
                [removable[i] for i in np.flatnonzero(harmful)], target)
        found[group] = row
    return found


def missed_weeds(evaluation: Evaluation, lists: dict[Session, Lists], multiplier: float,
                 min_plants: int) -> dict[str, dict]:
    """What a list costs a weed species it misses: in each session, each labelled weed species
    is left off its weed list in turn, with at least one other left on it. Per dataset and ALL,
    the shares of MISSED over those plants found, as a mean per species with at least
    `min_plants` of them, so a very common species doesn't decide the result."""
    calls = defaultdict(list)  # (group, class name): (right raw, right, crop raw, crop) per plant
    for session, found in lists.items():
        rows = evaluation.rows.get(session)
        if rows is None:
            continue
        for name in found.weed:
            plants = rows[evaluation.true[rows] == evaluation.classes.index(name)]
            rest = found.weed - {name}
            if not len(plants) or not rest:
                continue
            raw = evaluation.probs[plants]
            top = skew_probabilities(raw, evaluation.classes, evaluation.crops, found.crop, rest,
                                     use_lists=True, multiplier=multiplier).argmax(1)
            right = evaluation.classes.index(name)
            plant_calls = zip(raw.argmax(1) == right, top == right,
                              evaluation.is_crop[raw.argmax(1)], evaluation.is_crop[top])
            for call in plant_calls:
                calls[ALL, name].append(call)
                calls[session[0], name].append(call)
    means = defaultdict(list)
    for (group, _), plant_calls in calls.items():
        if len(plant_calls) >= min_plants:
            means[group].append(np.mean(plant_calls, axis=0))
    return {group: {"species": len(values), **dict(zip(
        ("right_unskewed", "right", "crop_unskewed", "crop"),
        np.mean(values, axis=0).tolist()))} for group, values in means.items()}


def list_sources(evaluation: Evaluation, settings: dict[str, Any]) -> dict[str, dict]:
    """The lists of each source in settings["sources"]."""
    first_pass = settings["first_pass"]
    makers = {"labelled": lambda: labelled_lists(evaluation),
              "first_pass": lambda: first_pass_lists(evaluation, first_pass["weed_share"],
                                                     first_pass["min_plants"]),
              "first_pass_paired": lambda: first_pass_lists(evaluation, first_pass["weed_share"],
                                                            first_pass["min_plants"], True)}
    return {source: makers[source]() for source in settings["sources"]}


def _sessions_with_list(evaluation: Evaluation, lists: dict[Session, Lists],
                        side: str) -> Counter:
    use_crop, use_weed = SIDES[side]
    counts = Counter()
    for session in evaluation.rows:
        found = lists.get(session)
        if found and ((use_crop and found.crop) or (use_weed and found.weed)):
            counts[session[0]] += 1
            counts[ALL] += 1
    return counts


def evaluate_run(evaluation: Evaluation, run: str, settings: dict[str, Any]) -> dict[str, list]:
    """The rows of a run's report: `scores` per case and dataset, `missed` for the `labelled`
    lists per multiplier, and `lists` with every session's lists per source."""
    sessions = Counter(session[0] for session in evaluation.rows)
    sessions[ALL] = len(evaluation.rows)
    rows = [{"run": run, "dataset": group, "source": "no skew", "lists": "", "multiplier": 1,
             "sessions": sessions[group], "sessions_with_list": 0, **values}
            for group, values in scores(evaluation, evaluation.probs, settings).items()]
    sources = list_sources(evaluation, settings)
    for source, lists in sources.items():
        for side in settings["lists"]:
            with_list = _sessions_with_list(evaluation, lists, side)
            for multiplier in settings["multipliers"]:
                probs = skewed(evaluation, lists, side, multiplier)
                rows += [{"run": run, "dataset": group, "source": source, "lists": side,
                          "multiplier": multiplier, "sessions": sessions[group],
                          "sessions_with_list": with_list[group], **values}
                         for group, values in scores(evaluation, probs, settings).items()]
    missed = []
    if "labelled" in sources:
        for multiplier in settings["multipliers"]:
            found = missed_weeds(evaluation, sources["labelled"], multiplier,
                                 settings["missed_min_plants"])
            missed += [{"run": run, "dataset": group, "multiplier": multiplier, **values}
                       for group, values in found.items()]
    counted = {session: len(found) for session, found in evaluation.rows.items()}
    lists = [{"run": run, "source": source, "dataset": session[0], "session": session[1],
              "plants": counted.get(session, 0), "partner": found.partner,
              "crop_list": " ".join(sorted(found.crop)), "weed_list": " ".join(sorted(found.weed))}
             for source, by_session in sources.items()
             for session, found in sorted(by_session.items())]
    return {"scores": rows, "missed": missed, "lists": lists}


def _spread(values: list[float | None], digits: int = 1) -> str:
    """Across runs: a percentage, or the range when the runs differ at this rounding."""
    known = [100 * value for value in values if value is not None]
    if not known:
        return "-"
    low, high = f"{min(known):.{digits}f}", f"{max(known):.{digits}f}"
    return f"{low}%" if low == high else f"{low}-{high}%"


def report(results: dict[str, list], settings: dict[str, Any], runs: list[str]) -> str:
    """The Markdown report: per dataset, one row per case, with the range across runs."""
    first_pass = settings["first_pass"]
    intro = (f"Percentages of the labelled plants found; a range is the spread across runs. "
             f"first_pass: top crop and weeds >= {first_pass['weed_share']:.0%} of weeds, "
             f"min {first_pass['min_plants']} plants. Lists: which of the two lists are used. "
             f"Species right leaves out other_plant, as by_dataset.csv does; class right counts "
             f"it. Compare decisions at the same crops removed, not at the rule file's "
             f"defer_below: a list raises the confidence. See docs/twostage.md (Species lists).")
    lines = [f"Runs: {', '.join(runs)}.", "", intro, ""]
    columns = list(SHARES) + [f"harmful_at_{t}" for t in settings["crops_removed"]]
    titles = [*SHARES.values(), *(f"harmful at <={t:.0%} crops" for t in settings["crops_removed"])]
    cases = defaultdict(list)
    for row in results["scores"]:
        cases[row["dataset"], row["source"], row["lists"], row["multiplier"]].append(row)
    for dataset in sorted({key[0] for key in cases}, key=lambda name: (name != ALL, name)):
        lines += [f"## {dataset}", "",
                  "| source | lists | x | sessions with a list | " + " | ".join(titles) + " |",
                  "|---" * (4 + len(titles)) + "|"]
        for key, rows in cases.items():
            if key[0] == dataset:
                with_list = sorted({row["sessions_with_list"] for row in rows})
                counted = (f"{with_list[0]}-{with_list[-1]}" if len(with_list) > 1
                           else str(with_list[0]))
                cells = [_spread([row[column] for row in rows]) for column in columns]
                lines.append(f"| {key[1]} | {key[2]} | {key[3]} | {counted}/{rows[0]['sessions']} "
                             f"| " + " | ".join(cells) + " |")
        lines.append("")
    if results["missed"]:
        intro = ("The session's labelled lists, each weed species left off in turn; mean per "
                 f"species with at least {settings['missed_min_plants']} plants found.")
        lines += ["## A weed species the list misses", "", intro, "",
                  "| dataset | x | species | " + " | ".join(MISSED.values()) + " |",
                  "|---" * (3 + len(MISSED)) + "|"]
        missed = defaultdict(list)
        for row in results["missed"]:
            missed[row["dataset"], row["multiplier"]].append(row)
        for (dataset, multiplier), rows in sorted(missed.items(),
                                                  key=lambda item: (item[0][0] != ALL, item[0])):
            species = sorted({row["species"] for row in rows})
            lines.append(f"| {dataset} | {multiplier} | {species[0]} | " + " | ".join(
                _spread([row[column] for row in rows]) for column in MISSED) + " |")
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(runs: list[Path], settings: dict[str, Any], stem: Path) -> str:
    """Evaluate the two-stage predictions of classifier runs. Writes <stem>.md, <stem>.csv (every
    number per run), <stem>_missed.csv and <stem>_lists.csv (every session's lists, to trace or
    redo a result); returns the Markdown."""
    results = {"scores": [], "missed": [], "lists": []}
    ground_truths = {}
    for run in runs:
        folder = twostage_eval(resolve(settings["twostage_dir"]), run, settings["split"],
                               settings["checkpoint"])
        if folder is None:
            raise FileNotFoundError(f"No two-stage evaluation of {run.name}, {settings['split']} "
                                    f"{settings['checkpoint']}; run scripts/predict_twostage.py")
        found = evaluate_run(load_evaluation(folder, ground_truths), run.name, settings)
        for key, rows in found.items():
            results[key] += rows
    text = report(results, settings, [run.name for run in runs])
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    _write_csv(stem.with_suffix(".csv"), results["scores"])
    _write_csv(stem.parent / f"{stem.name}_missed.csv", results["missed"])
    _write_csv(stem.parent / f"{stem.name}_lists.csv", results["lists"])
    return text
