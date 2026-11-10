"""Detector runs side by side: their evaluations (by_dataset.csv of each split that
evaluate_detector.py evaluated) and scale tests (report.csv of evaluate_scale.py), as one
Markdown file. scripts/run_detector.py writes it after its detector variants.
"""

import csv
import json
from pathlib import Path
from typing import Any

from weedseg.detector.run import eval_dir, evaluated_splits, resolve_run_dir, scale_dir

ALL = "all"


def read_csv(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV file, or [] when it doesn't exist."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_epoch(run_dir: Path) -> dict[str, Any] | None:
    """The epoch line of metrics.jsonl that saved best.pt (the last one marked best)."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("best"):
                best = record
    return best


def _number(value: str | None, scale: float = 1, digits: int = 1) -> str:
    if value in (None, ""):
        return "-"
    return f"{float(value) * scale:.{digits}f}"


def _per_image(row: dict[str, str] | None) -> str:
    if not row or not row.get("images"):
        return "-"
    return f"{int(row['detections_on_no_plant']) / int(row['images']):.2f}"


def write_comparison(configs: list[dict[str, Any]], stem: Path) -> str:
    """Compare the runs of detector `configs` (each run at its `evaluate.checkpoint`); write
    <stem>.md and return its text."""
    names = [resolve_run_dir(config).name for config in configs]
    lines = ["# Detector runs compared", ""]
    lines += ["| run | training short side (px) | best epoch | val box AP | val box AP50 |",
              "|---|---|---|---|---|"]
    for name, config in zip(names, configs):
        sizes = config["model"]["min_size"]
        trained = str(sizes[0]) if len(sizes) == 1 else f"{min(sizes)}-{max(sizes)}"
        best = best_epoch(resolve_run_dir(config))
        lines.append(f"| {name} | {trained} | {best['epoch'] if best else '-'} | "
                     f"{_number(best and best.get('val_AP'), digits=3)} | "
                     f"{_number(best and best.get('val_AP50'), digits=3)} |")
    lines += ["", ("Best epoch and val numbers from training (metrics.jsonl); the tables below "
                   "from the evaluation of each run's checkpoint."), ""]

    for split in evaluated_splits(configs[0]):
        tables = [{row["dataset"]: row for row in read_csv(eval_dir(c, split) / "by_dataset.csv")}
                  for c in configs]
        datasets = [ALL, *sorted({d for t in tables for d in t} - {ALL})]
        lines += [f"## Evaluation: {split}", "",
                  ("Plants found (%, score >= the evaluation's threshold), box AP50, and "
                   "detections on no labelled plant per image; one column per run."), "",
                  "| dataset | " + " | ".join(f"found, {n}" for n in names) + " | "
                  + " | ".join(f"AP50, {n}" for n in names) + " | "
                  + " | ".join(f"false/image, {n}" for n in names) + " |",
                  "|---|" + "---|" * (3 * len(names))]
        for dataset in datasets:
            rows = [t.get(dataset) for t in tables]
            lines.append(f"| {dataset} | "
                         + " | ".join(_number(r and r["found"], 100) for r in rows) + " | "
                         + " | ".join(_number(r and r["AP50"], digits=3) for r in rows) + " | "
                         + " | ".join(_per_image(r) for r in rows) + " |")
        lines.append("")

    reports = [read_csv(scale_dir(c) / "report.csv") for c in configs]
    datasets = sorted({row["dataset"] for report in reports for row in report})
    cases = list(dict.fromkeys((row["variant"], row["target"])
                               for report in reports for row in report))
    lines += ["## Scale test", "",
              "Plants found (%) at each scale (mm/px at the detector; native = each image's "
              "own), cells: " + " / ".join(names) + ". See each run's scale/*/report.md for "
              "the rest.", ""]
    if not datasets:
        lines += ["No scale test found.", ""]
    else:
        lines += ["| case | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
        for variant, target in cases:
            cells = []
            for dataset in datasets:
                values = []
                for report in reports:
                    row = next((r for r in report if r["dataset"] == dataset
                                and r["variant"] == variant and r["target"] == target), None)
                    values.append(_number(row and row.get("found"), 100))
                cells.append(" / ".join(values))
            label = {"as_is": "", "deployed": "deployed resize, ",
                     "rescaled": "rescaled from "}[variant] + target
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    return text
