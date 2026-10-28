"""Classifier runs side by side: the evaluation of each (scripts/evaluate_classifier.py) and the
two-stage prediction made with it (scripts/predict_twostage.py), on the same split.

The rows are the settings in which the runs differ, then the overall numbers (classifier
accuracy overall and per dataset, and both stages' plants found, species right and decisions),
then, with target shares of crops removed, the harmful weeds removed at each (the decisions
at one crop safety, with the defer_below that gives it), then those that show how much a run
relies on the look of a dataset (cross-dataset accuracy, fit of the partial labels). A
missing file leaves "-". Runs whose settings differ only in `seed` are pointed out: their
gaps show the run-to-run noise. See docs/twostage.md.
"""

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import yaml

from weedseg.config import load_config
from weedseg.decisions import harmful_removed_at, load_rules, removed_up_to, role_of
from weedseg.eppo import load_taxa
from weedseg.evaluation import ignore_category_ids, match_detections
from weedseg.maskrcnn.data import load_masked_coco
from weedseg.maskrcnn.run import resolve_run_dir
from weedseg.paths import resolve

# Columns of the two-stage by_dataset.csv shown, with their row names.
TWOSTAGE_COLUMNS = {"found": "plants found", "species_right": "species right (of found)",
                    "crops_removed": "crops removed", "harmful_removed": "harmful removed",
                    "harmful_left": "harmful left"}
# Settings that differ between any two runs and say nothing about them.
NAMING = ("run_name",)


def run_dir(entry: str) -> Path:
    """A classifier run folder, or the run of a classifier config (a .yaml file)."""
    if entry.endswith((".yaml", ".yml")):
        return resolve_run_dir(load_config(entry))
    return resolve(entry)


def twostage_eval(twostage_dir: Path, run: Path, split: str, checkpoint: str) -> Path | None:
    """The newest two-stage evaluation made with that checkpoint of the classifier run, as
    predict_twostage.py names it with the default run_name:
    <twostage_dir>/<detector run>__<run>/eval/<split>_<checkpoint>. Folders made up to
    2026-10-26 are named <split> and were always made with best.pt."""
    patterns = [f"*__{run.name}/eval/{split}_{checkpoint}"]
    if checkpoint == "best":
        patterns.append(f"*__{run.name}/eval/{split}")
    found = [path for pattern in patterns for path in twostage_dir.glob(pattern)
             if (path / "by_dataset.csv").exists()]
    return max(found, key=lambda path: (path / "by_dataset.csv").stat().st_mtime, default=None)


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """{"section.key": value as text} of a nested config."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = json.dumps(value) if isinstance(value, list) else str(value)
    return flat


def comparison(runs: list[Path], split: str, checkpoint: str, twostage_dir: Path,
               crops_removed: list[float] = ()) -> tuple[list[str], list[list[str]]]:
    """Notes (which run is which, seed-only pairs) and the table rows: a name, then one value
    per run. Classifier evaluations are read from <run>/eval/<split>_<checkpoint>.
    `crops_removed`: target shares of crops removed for `decisions_at_crops_removed`."""
    letters = ascii_uppercase[:len(runs)]
    notes = []
    settings, classifier, twostage, same_crops = [], [], [], []
    ground_truths = {}
    for letter, run in zip(letters, runs):
        found = twostage_eval(twostage_dir, run, split, checkpoint)
        notes.append(f"{letter}: {run.name}" + (f" (both stages: {found.parent.parent.name})"
                                                 if found else " (no two-stage evaluation)"))
        config = run / "config.yaml"
        settings.append(flatten(yaml.safe_load(config.read_text(encoding="utf-8")) or {})
                        if config.exists() else {})
        classifier.append(read_classifier_eval(run, run / "eval" / f"{split}_{checkpoint}",
                                               checkpoint))
        twostage.append(read_csv_by(found / "by_dataset.csv", "dataset") if found else {})
        same_crops.append({})
        if found and crops_removed:
            try:
                same_crops[-1] = decisions_at_crops_removed(found, crops_removed, ground_truths)
            except (FileNotFoundError, ValueError) as error:
                notes.append(f"{letter}: no decisions at the same crops removed ({error})")

    rows = []
    present = [s for s in settings if s]  # a run not trained yet differs in nothing
    differing = sorted(key for key in set().union(*present) - set(NAMING)
                       if len({s.get(key, "-") for s in present}) > 1)
    if differing:
        rows.append(["**Settings that differ**", *[""] * len(runs)])
        rows += [[key, *(s.get(key, "-") for s in settings)] for key in differing]
    for i, j in combinations(range(len(runs)), 2):
        if settings[i] and settings[j] and settings[i].get("seed") != settings[j].get(
                "seed") and all(
                settings[i].get(key) == settings[j].get(key) for key in differing if key != "seed"):
            notes.append(f"{letters[i]} and {letters[j]} differ only in seed: their gaps show "
                         "how much results vary by chance")

    rows.append([f"**Classifier alone, {split} crops**", *[""] * len(runs)])
    rows.append([f"epoch evaluated ({checkpoint})",
                 *(str(c["epoch"]) if c["epoch"] is not None else "-" for c in classifier)])
    rows.append(["val accuracy, best epoch minus last",
                 *(share(c["last_below_best"]) for c in classifier)])
    for key, name in (("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy")):
        rows.append([name, *(share(c["metrics"].get(key)) for c in classifier)])
    for dataset in sorted(set().union(*(c["by_dataset"] for c in classifier))):
        for key, name in (("accuracy", "accuracy"), ("balanced_accuracy", "balanced")):
            rows.append([f"{name}, {dataset}",
                         *(share(c["by_dataset"].get(dataset, {}).get(key)) for c in classifier)])

    rows.append([f"**Both stages, {split} images**", *[""] * len(runs)])
    for dataset in sorted(set().union(*twostage), key=lambda d: (d != "all", d)):
        for key, name in TWOSTAGE_COLUMNS.items():
            rows.append([f"{name}, {dataset}",
                         *(share(t.get(dataset, {}).get(key)) for t in twostage)])

    if crops_removed:
        rows.append(["**Both stages, harmful removed at the same crops removed (defer_below)**",
                     *[""] * len(runs)])
    for dataset in sorted(set().union(*same_crops), key=lambda d: (d != "all", d)):
        for target in crops_removed:
            cells = [removed_cell(*s.get(dataset, {}).get(target, (None, None)))
                     for s in same_crops]
            if any(cell != "-" for cell in cells):
                rows.append([f"harmful removed, up to {share(target)} of crops, {dataset}",
                             *cells])

    rows.append(["**Reliance on the dataset**", *[""] * len(runs)])
    rows.append(["cross-dataset accuracy",
                 *(share(c["metrics"].get("cross_dataset_accuracy")) for c in classifier)])
    rows.append(["partial labels fit",
                 *(share(c["metrics"].get("partial_label_fit")) for c in classifier)])
    for dataset in sorted(set().union(*(c["partial_fit"] for c in classifier))):
        rows.append([f"partial labels fit, {dataset}",
                     *(share(c["partial_fit"].get(dataset)) for c in classifier)])
    return notes, [["", *letters], *rows]


def read_classifier_eval(run: Path, folder: Path, checkpoint: str) -> dict[str, Any]:
    """What `comparison` shows of a classifier run: the epoch of `checkpoint` (best or
    last), how far the last epoch's val accuracy is below the best's, metrics.json,
    accuracy_by_dataset.csv and the partial labels' fit per dataset."""
    epoch = last_below_best = None
    if (run / "metrics.jsonl").exists():
        records = [json.loads(line) for line
                   in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if records:
            best = max(records, key=lambda r: r["val_accuracy"])
            epoch = {"best": best["epoch"], "last": records[-1]["epoch"]}.get(checkpoint)
            last_below_best = best["val_accuracy"] - records[-1]["val_accuracy"]
    metrics = (json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
               if (folder / "metrics.json").exists() else {})
    crops, fitting = {}, {}
    for row in read_csv_by(folder / "partial_labels.csv", None):
        crops[row["dataset"]] = crops.get(row["dataset"], 0) + int(row["crops"])
        fitting[row["dataset"]] = (fitting.get(row["dataset"], 0)
                                   + float(row["label_fit"]) * int(row["crops"]))
    return {"epoch": epoch, "last_below_best": last_below_best, "metrics": metrics,
            "by_dataset": read_csv_by(folder / "accuracy_by_dataset.csv", "dataset"),
            "partial_fit": {name: fitting[name] / crops[name] for name in crops}}


def decisions_at_crops_removed(found: Path, targets: list[float],
                               ground_truths: dict[Path, dict]) -> dict[str, dict[float, tuple]]:
    """Per dataset of a two-stage evaluation folder (and "all"), for each target share of
    crops removed: the share of harmful plants removed and the defer_below
    (`harmful_removed_at`). Made again from the folder's predictions.json, with the ground
    truth, matching and rule file of its evaluation.json; the rule file as it is now, so all
    runs get the same roles. `ground_truths` keeps each ground truth loaded, by path. Raises
    FileNotFoundError or ValueError when it can't be done."""
    evaluation = json.loads((found / "evaluation.json").read_text(encoding="utf-8"))
    settings = evaluation["settings"]
    if settings.get("max_images") is not None:
        raise ValueError(f"{found} was evaluated on {settings['max_images']} images only")
    if not settings.get("rules"):
        raise ValueError(f"{found} was evaluated without a rule file")
    path = resolve(evaluation["split_file"])
    if path not in ground_truths:
        ground_truths[path] = load_masked_coco(path)
    gt = ground_truths[path]
    results = json.loads((found / "predictions.json").read_text(encoding="utf-8"))
    taxa = load_taxa(resolve(settings["eppo_taxa"]))
    rules = load_rules(resolve(settings["rules"]), taxa)
    ignore_ids = ignore_category_ids(gt)
    names = {c["id"]: c["name"] for c in gt["categories"] if c["id"] not in ignore_ids}
    roles = {name: role_of(name, rules, taxa) for name in names.values()}
    dataset_of = {image["id"]: image.get("dataset") or "(none)" for image in gt["images"]}
    plants = defaultdict(lambda: {"crop": [], "harmful": []})
    for match in match_detections(gt, results, None, settings["score_threshold"],
                                  settings["iou_threshold"], settings["match_by"]):
        role = roles[names[match.plant["category_id"]]] if match.plant else None
        if role in ("crop", "harmful"):
            value = (removed_up_to(match.detection, names, roles, rules.plant_defer_below)
                     if match.detection else None)
            for group in ("all", dataset_of[match.image_id]):
                plants[group][role].append(value)
    return {group: {target: harmful_removed_at(values["crop"], values["harmful"], target)
                    for target in targets} for group, values in plants.items()}


def removed_cell(harmful: float | None, defer_below: float | None) -> str:
    """A share of harmful plants removed and, in brackets, the defer_below it is made at."""
    return "-" if harmful is None else f"{share(harmful)} ({defer_below:.3f})"


def read_csv_by(path: Path, key: str | None) -> Any:
    """Rows of a CSV file ([] or {} when it doesn't exist): a list, or {row[key]: row}."""
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return rows if key is None else {row[key]: row for row in rows}


def share(value: Any) -> str:
    """A share as a percentage; two decimals under 1%, where e.g. crops removed lie."""
    if value in (None, ""):
        return "-"
    value = float(value)
    return f"{value:.2%}" if 0 < value < 0.01 else f"{value:.1%}"


def write_comparison(runs: list[Path], split: str, checkpoint: str, twostage_dir: Path,
                     stem: Path, crops_removed: list[float] = ()) -> str:
    """The comparison as <stem>.md (notes and table) and <stem>.csv; returns the Markdown."""
    notes, table = comparison(runs, split, checkpoint, twostage_dir, crops_removed)
    lines = [*notes, "", "| " + " | ".join(table[0]) + " |",
             "|---" * len(table[0]) + "|", *("| " + " | ".join(row) + " |" for row in table[1:])]
    text = "\n".join(lines) + "\n"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["", *(run.name for run in runs)])
        writer.writerows(row for row in table[1:] if not row[0].startswith("**"))
    return text
