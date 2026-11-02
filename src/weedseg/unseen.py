"""Classifier runs on plants of datasets they never trained on, next to runs that did.

Every dataset's val split shares sites and days with its train split, so the usual evaluation
can't show what a model does at a new site, nor what more diverse training data buys. Here
runs of different combinations score the same plants: a dataset new to one run is known to
another. Plants are grouped by dataset, split, class and site. Plants of a species the run has
no class for are scored too (what does it do with them?), and partly labelled ones by the
probability of the classes their label rules out, not by the top class alone: on a plant of
an unknown species the output is often flat, and a top class at 0.05 says little.

scripts/evaluate_unseen.py runs it. See docs/twostage.md (Unseen sites).
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import numpy as np

from weedseg.compare import share
from weedseg.data.coco import OTHER_CATEGORY
from weedseg.data.combine import describe_classes
from weedseg.decisions import ACTIONS

ALL_SITES = "all sites"
GroupKey = tuple[str, str, str, str]  # dataset, split, plant label, site


def trained_on(data_dir: Path) -> set[str]:
    """The datasets of a species data folder: the first line of its classes.txt (combinations
    since 2026-10-30), else the name of the folder above, the datasets joined by +."""
    report = data_dir / "classes.txt"
    if report.exists():
        first = report.read_text(encoding="utf-8").splitlines()[0]
        if first.startswith("Datasets: "):
            return set(first.removeprefix("Datasets: ").split(", "))
    return set(data_dir.parent.name.split("+"))


def scored_on(split: str, dataset: str, run_datasets: set[str]) -> bool:
    """A run scores a dataset's val split whether it trained on it or not, for the comparison
    on the same plants, and any other split only if it never trained on that dataset."""
    return split == "val" or dataset not in run_datasets


def subset(coco: dict[str, Any], datasets: list[str]) -> dict[str, Any]:
    """A COCO file with only the images of these datasets and their annotations."""
    images = [image for image in coco["images"] if image.get("dataset") in datasets]
    ids = {image["id"] for image in images}
    return {**coco, "images": images,
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}


def site_of(image: dict[str, Any], patterns: dict[str, str]) -> str | None:
    """The site in an image's group, by the regular expression of its dataset; None when the
    dataset has none (the sites aren't known) or the group doesn't match."""
    pattern = patterns.get(image.get("dataset"))
    found = re.search(pattern, image.get("group", "")) if pattern else None
    return found.group(0) if found else None


def plant_label(row: dict[str, Any], data_classes: list[str]) -> str:
    """What a plant is: its class, or for a partial label its code and what the label allows."""
    if row["allowed"]:
        return f"{row['eppo']}: {describe_classes(row['allowed'], data_classes)}"
    return row["category"]


def actions(probs: np.ndarray, classes: list[str], roles: dict[str, str],
            defer_below: float) -> np.ndarray:
    """What weedseg.decisions.decide does with each crop, given that it is a plant: keep or
    remove when the summed probability of that action is at least `defer_below`, else defer."""
    def summed(action: str) -> np.ndarray:
        return probs[:, [i for i, name in enumerate(classes)
                         if ACTIONS[roles[name]] == action]].sum(1)

    keep, remove = summed("keep"), summed("remove")
    return np.where(np.maximum(keep, remove) >= defer_below,
                    np.where(remove > keep, "remove", "keep"), "defer")


def score_plants(rows: list[dict[str, Any]], probs: np.ndarray, classes: list[str],
                 roles: dict[str, str], taxa: dict[str, dict[str, str]], defer_below: float,
                 sure: float) -> dict[str, Any]:
    """The numbers of one group of plants (one label) for one run. `right` and `right_genus`
    are for labelled plants, `right` None when the run has no class for them; `ruled_out_top`
    and `ruled_out_sure` for partly labelled ones: the share whose top class the label rules
    out, and whose ruled-out classes together reach `sure`."""
    top = probs.argmax(1)
    names = np.array(classes)[top]
    action = actions(probs, classes, roles, defer_below)
    counts = Counter(names.tolist()).most_common(2)
    scores = {
        "plants": len(rows), "sure": float((probs.max(1) >= sure).mean()),
        "median_top": float(np.median(probs.max(1))),
        "other_plant": float((names == OTHER_CATEGORY).mean()),
        "keep": float((action == "keep").mean()), "remove": float((action == "remove").mean()),
        "defer": float((action == "defer").mean()),
        "most_predicted": "; ".join(f"{name} {n / len(rows):.0%}" for name, n in counts),
    }
    allowed = rows[0]["allowed"]
    if allowed:
        ruled_out = [i for i, name in enumerate(classes) if name not in allowed]
        return scores | {"ruled_out_top": float(np.isin(top, ruled_out).mean()),
                         "ruled_out_sure": float((probs[:, ruled_out].sum(1) >= sure).mean())}
    category = rows[0]["category"]
    genus = taxa.get(category, {}).get("genus")
    return scores | {
        "right": float((names == category).mean()) if category in classes else None,
        "right_genus": float(np.mean([bool(genus) and name in taxa
                                      and taxa[name]["genus"] == genus for name in names])),
    }


def score_groups(rows: list[dict[str, Any]], probs: np.ndarray, classes: list[str],
                 roles: dict[str, str], taxa: dict[str, dict[str, str]], defer_below: float,
                 sure: float, data_classes: list[str]) -> dict[GroupKey, dict[str, Any]]:
    """`score_plants` per dataset, split, plant label and site. Rows have `split` and `site`
    (None = sites not known); a dataset with sites also gets all of them together."""
    members = defaultdict(list)
    for i, row in enumerate(rows):
        key = row["dataset"], row["split"], plant_label(row, data_classes)
        members[(*key, ALL_SITES)].append(i)
        if row["site"]:
            members[(*key, row["site"])].append(i)
    return {key: score_plants([rows[i] for i in positions], probs[positions], classes, roles,
                              taxa, defer_below, sure)
            for key, positions in members.items()}


def group_order(key: GroupKey) -> tuple:
    """Dataset, then val before other splits, then label, then all sites before each site."""
    dataset, split, label, site = key
    return dataset, split != "val", split, label, site != ALL_SITES, site


def table_rows(runs: list[dict[str, Any]], sure: float) -> list[list[str]]:
    """The header and rows of the table. `runs`: per run `datasets` (trained on) and `scores`
    (`score_groups` of its splits). A group's first row tells which runs were new to that
    dataset; "-" where a run didn't score the group or has no such number."""
    letters = ascii_uppercase[:len(runs)]

    def values(key: GroupKey, name: str) -> list[str]:
        return [share(run["scores"].get(key, {}).get(name)) for run in runs]

    table = [["", *letters]]
    for key in sorted({key for run in runs for key in run["scores"]}, key=group_order):
        dataset, split, label, site = key
        plants = next(run["scores"][key]["plants"] for run in runs if key in run["scores"])
        table.append([f"**{dataset} {split}, {label}, {site}: {plants} plants**",
                      *("seen" if dataset in run["datasets"] else "new" for run in runs)])
        partly = any("ruled_out_top" in run["scores"].get(key, {}) for run in runs)
        rows = ([("top class ruled out", "ruled_out_top"),
                 (f"ruled-out classes reach {sure:g}", "ruled_out_sure")] if partly else
                [("right", "right"), ("right genus", "right_genus")])
        table += [[name, *values(key, column)] for name, column in rows]
        table.append([f"top class at {sure:g} or more", *values(key, "sure")])
        table.append(["other_plant", *values(key, "other_plant")])
        table.append(["keep / remove / defer", *(
            " / ".join(share(run["scores"][key][a]) for a in ("keep", "remove", "defer"))
            if key in run["scores"] else "-" for run in runs)])
        table.append(["median top probability", *(
            f"{run['scores'][key]['median_top']:.2f}" if key in run["scores"] else "-"
            for run in runs)])
        table.append(["most predicted", *(run["scores"].get(key, {}).get("most_predicted", "-")
                                          for run in runs)])
    return table


def write_results(runs: list[dict[str, Any]], sure: float, defer_below: float, stem: Path) -> str:
    """<stem>.md (which run is which, and the table) and <stem>.csv (one row per run and group,
    with `seen`: whether the run trained on that dataset); returns the Markdown."""
    letters = ascii_uppercase[:len(runs)]
    notes = [f"{letter}: {run['name']} ({len(run['classes'])} classes; trained on "
             f"{', '.join(sorted(run['datasets']))})" for letter, run in zip(letters, runs)]
    notes.append(f"Classifier alone, on labelled boxes; keep / remove / defer at defer_below "
                 f"{defer_below:g}. seen = the run trained on that dataset, new = it never did.")
    table = table_rows(runs, sure)
    lines = [*notes, "", "| " + " | ".join(table[0]) + " |", "|---" * len(table[0]) + "|",
             *("| " + " | ".join(row) + " |" for row in table[1:])]
    text = "\n".join(lines) + "\n"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
    columns = ["plants", "right", "right_genus", "ruled_out_top", "ruled_out_sure", "sure",
               "other_plant", "keep", "remove", "defer", "median_top", "most_predicted"]
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "letter", "dataset", "split", "plant", "site", "seen", *columns])
        for letter, run in zip(letters, runs):
            for key in sorted(run["scores"], key=group_order):
                writer.writerow([run["name"], letter, *key, key[0] in run["datasets"],
                                 *(run["scores"][key].get(column) for column in columns)])
    return text
