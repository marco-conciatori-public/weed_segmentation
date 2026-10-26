"""Classifier runs side by side: the evaluation of each (scripts/evaluate_classifier.py) and the
two-stage prediction made with it (scripts/predict_twostage.py), on the same split.

The rows are the settings in which the runs differ, then the overall numbers (classifier
accuracy overall and per dataset, and both stages' plants found, species right and decisions),
then those that show how much a run relies on the look of a dataset (cross-dataset accuracy,
fit of the partial labels). A missing file leaves "-". Runs whose settings differ only in
`seed` are pointed out: their gaps show the run-to-run noise. See docs/twostage.md.
"""

import csv
import json
from itertools import combinations
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import yaml

from weedseg.config import load_config
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


def twostage_eval(twostage_dir: Path, run: Path, split: str) -> Path | None:
    """The newest two-stage evaluation made with the classifier run, as predict_twostage.py
    names it with the default run_name: <twostage_dir>/<detector run>__<run>/eval/<split>."""
    found = [path for path in twostage_dir.glob(f"*__{run.name}/eval/{split}")
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


def comparison(runs: list[Path], split: str, checkpoint: str,
               twostage_dir: Path) -> tuple[list[str], list[list[str]]]:
    """Notes (which run is which, seed-only pairs) and the table rows: a name, then one value
    per run. Classifier evaluations are read from <run>/eval/<split>_<checkpoint>."""
    letters = ascii_uppercase[:len(runs)]
    notes = []
    settings, classifier, twostage = [], [], []
    for letter, run in zip(letters, runs):
        found = twostage_eval(twostage_dir, run, split)
        notes.append(f"{letter}: {run.name}" + (f" (both stages: {found.parent.parent.name})"
                                                 if found else " (no two-stage evaluation)"))
        config = run / "config.yaml"
        settings.append(flatten(yaml.safe_load(config.read_text(encoding="utf-8")) or {})
                        if config.exists() else {})
        classifier.append(read_classifier_eval(run, run / "eval" / f"{split}_{checkpoint}"))
        twostage.append(read_csv_by(found / "by_dataset.csv", "dataset") if found else {})

    rows = []
    differing = sorted(key for key in set().union(*settings) - set(NAMING)
                       if len({s.get(key, "-") for s in settings}) > 1)
    if differing:
        rows.append(["**Settings that differ**", *[""] * len(runs)])
        rows += [[key, *(s.get(key, "-") for s in settings)] for key in differing]
    for i, j in combinations(range(len(runs)), 2):
        if settings[i].get("seed") != settings[j].get("seed") and all(
                settings[i].get(key) == settings[j].get(key) for key in differing if key != "seed"):
            notes.append(f"{letters[i]} and {letters[j]} differ only in seed: their gaps show "
                         "how much results vary by chance")

    rows.append([f"**Classifier alone, {split} crops**", *[""] * len(runs)])
    rows.append(["best epoch", *(str(c["epoch"]) if c["epoch"] is not None else "-"
                                 for c in classifier)])
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

    rows.append(["**Reliance on the dataset**", *[""] * len(runs)])
    rows.append(["cross-dataset accuracy",
                 *(share(c["metrics"].get("cross_dataset_accuracy")) for c in classifier)])
    rows.append(["partial labels fit",
                 *(share(c["metrics"].get("partial_label_fit")) for c in classifier)])
    for dataset in sorted(set().union(*(c["partial_fit"] for c in classifier))):
        rows.append([f"partial labels fit, {dataset}",
                     *(share(c["partial_fit"].get(dataset)) for c in classifier)])
    return notes, [["", *letters], *rows]


def read_classifier_eval(run: Path, folder: Path) -> dict[str, Any]:
    """What `comparison` shows of a classifier run: its best epoch, metrics.json,
    accuracy_by_dataset.csv and the partial labels' fit per dataset."""
    epoch = None
    if (run / "metrics.jsonl").exists():
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("best"):
                epoch = record["epoch"]
    metrics = (json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
               if (folder / "metrics.json").exists() else {})
    crops, fitting = {}, {}
    for row in read_csv_by(folder / "partial_labels.csv", None):
        crops[row["dataset"]] = crops.get(row["dataset"], 0) + int(row["crops"])
        fitting[row["dataset"]] = (fitting.get(row["dataset"], 0)
                                   + float(row["label_fit"]) * int(row["crops"]))
    return {"epoch": epoch, "metrics": metrics,
            "by_dataset": read_csv_by(folder / "accuracy_by_dataset.csv", "dataset"),
            "partial_fit": {name: fitting[name] / crops[name] for name in crops}}


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
                     stem: Path) -> str:
    """The comparison as <stem>.md (notes and table) and <stem>.csv; returns the Markdown."""
    notes, table = comparison(runs, split, checkpoint, twostage_dir)
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
