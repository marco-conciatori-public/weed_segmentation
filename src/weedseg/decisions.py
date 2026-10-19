"""What the weeding robot does with each recognised plant: keep, remove or defer.

The model recognises species; a rule file (see configs/rules/example.yaml) gives each a role:
crop or neutral -> keep, harmful -> remove. `other_plant` (species the model doesn't know) is
deferred to a generic model, and so is any detection with a confidence under `defer_below`.
Rule keys are EPPO codes at any rank (species, genus, family, order); the most specific key
that matches wins.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from weedseg.data.coco import OTHER_CATEGORY
from weedseg.eppo import PARENT_RANKS

ROLES = ("crop", "harmful", "neutral")
ACTIONS = {"crop": "keep", "neutral": "keep", "harmful": "remove", OTHER_CATEGORY: "defer"}
TRUE_ROWS = [*ROLES, OTHER_CATEGORY, "no plant"]  # "no plant": detections on no labelled plant
ACTION_COLUMNS = ["keep", "remove", "defer", "missed"]


@dataclass
class Rules:
    default: str
    rules: dict[str, str]  # EPPO code -> role
    defer_below: float | None = None  # detections less confident than this are deferred


def load_rules(path: Path, taxa: dict[str, dict[str, str]]) -> Rules:
    """Read a rule file; roles must be in ROLES and codes known to the EPPO taxa table."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rules = Rules(default=data["default"], rules=dict(data.get("rules") or {}),
                  defer_below=data.get("defer_below"))
    if rules.defer_below is not None and not 0 <= rules.defer_below <= 1:
        raise ValueError(f"defer_below in {path} must be between 0 and 1")
    bad_roles = {role for role in [rules.default, *rules.rules.values()] if role not in ROLES}
    if bad_roles:
        raise ValueError(f"Unknown roles in {path}: {sorted(bad_roles)}; use {', '.join(ROLES)}")
    known = set(taxa) | {row[rank] for row in taxa.values() for rank in PARENT_RANKS}
    unknown = sorted(set(rules.rules) - known)
    if unknown:
        raise ValueError(f"EPPO codes in {path} not in the taxa table: {', '.join(unknown)}")
    return rules


def role_of(name: str, rules: Rules, taxa: dict[str, dict[str, str]]) -> str:
    """Role of a model class (an EPPO code, or OTHER_CATEGORY, which keeps its own name)."""
    if name == OTHER_CATEGORY:
        return OTHER_CATEGORY
    if name not in taxa:
        raise ValueError(f"Class {name!r} is not an EPPO code; decision rules need the "
                         "species-level classes (taxonomy species)")
    lineage = [name, *(taxa[name][rank] for rank in PARENT_RANKS)]  # most specific first
    return next((rules.rules[code] for code in lineage if code in rules.rules), rules.default)


def defer_uncertain(results: list[dict[str, Any]], categories: list[dict[str, Any]],
                    defer_below: float | None) -> list[dict[str, Any]]:
    """Predictions (COCO results) with those under `defer_below` relabelled OTHER_CATEGORY,
    so they count as deferred. Detection matching ignores classes, so it is unaffected."""
    if defer_below is None:
        return results
    other_id = next((c["id"] for c in categories if c["name"] == OTHER_CATEGORY), None)
    if other_id is None:
        raise ValueError(f"defer_below needs an {OTHER_CATEGORY} category")
    return [{**r, "category_id": other_id} if r["score"] < defer_below else r for r in results]


def decision_matrix(matrix: np.ndarray, names: list[str], roles: dict[str, str]) -> np.ndarray:
    """Group a class confusion matrix (see evaluation.confusion_matrix: rows ground truth,
    columns predictions, last row/column background) by role and action.

    Rows follow TRUE_ROWS (the true role, from the ground-truth class), columns
    ACTION_COLUMNS (the action from the predicted class; "missed" = not detected).
    """
    *classes, _ = names
    row_of = [TRUE_ROWS.index(roles[name]) for name in classes] + [TRUE_ROWS.index("no plant")]
    column_of = ([ACTION_COLUMNS.index(ACTIONS[roles[name]]) for name in classes]
                 + [ACTION_COLUMNS.index("missed")])
    grouped = np.zeros((len(TRUE_ROWS), len(ACTION_COLUMNS)), np.int64)
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            grouped[row_of[i], column_of[j]] += count
    return grouped


def decision_summary(grouped: np.ndarray) -> dict[str, Any]:
    """Shares per true role (None when the role has no plants) and the count of removals
    where no plant is labelled."""
    def share(role: str, *actions: str) -> float | None:
        row = grouped[TRUE_ROWS.index(role)]
        total = row.sum()
        return (float(sum(row[ACTION_COLUMNS.index(a)] for a in actions) / total)
                if total else None)

    return {
        "crops_removed": share("crop", "remove"),
        "harmful_removed": share("harmful", "remove"),
        "harmful_left": share("harmful", "keep", "missed"),
        "harmful_deferred": share("harmful", "defer"),
        "neutral_removed": share("neutral", "remove"),
        "other_plant_deferred": share(OTHER_CATEGORY, "defer"),
        "removals_on_no_plant": int(grouped[TRUE_ROWS.index("no plant"),
                                            ACTION_COLUMNS.index("remove")]),
    }
