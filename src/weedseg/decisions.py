"""What the weeding robot does with each recognised plant: keep, remove or defer.

The model recognises species; a rule file (see configs/rules/example.yaml) gives each a role:
crop or neutral -> keep, harmful -> remove. `other_plant` (species the model doesn't know) is
deferred to a generic model. Rule keys are EPPO codes at any rank (species, genus, family,
order); the most specific key that matches wins.

With `defer_below`, the robot acts on its confidence in the action, not in the species: the
probabilities of all classes leading to one action are summed (see `decide`). A plant the
model can't place between two harmful grasses is still removed, and only plants whose action
is unsure go to the generic model.

`level_accuracy` tells, for the plants the model found, at which taxonomic level its class is
still right (a wrong species may have the right genus, family or role).
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
LEVELS = ("species", *PARENT_RANKS, "role")


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


def action_scores(detection: dict[str, Any], names: dict[int, str],
                  roles: dict[str, str]) -> dict[str, float]:
    """Probability of each action for a detection (a COCO result; `names` maps category ids
    to class names): the sum of its `class_scores` over the classes leading to that action.
    Without `class_scores` (predictions made before they existed), only the detection's own
    class and score count."""
    class_scores = (detection.get("class_scores")
                    or {detection["category_id"]: detection["score"]})
    totals = dict.fromkeys(("keep", "remove", "defer"), 0.0)
    for category_id, probability in class_scores.items():
        totals[ACTIONS[roles[names[int(category_id)]]]] += probability
    return totals


def decide(detection: dict[str, Any], names: dict[int, str], roles: dict[str, str],
           defer_below: float | None) -> str:
    """What the robot does with a detection: keep, remove or defer.

    Without `defer_below`, the action of the detection's class. With it, keep or remove
    when the summed probability of that action (`action_scores`) is at least `defer_below`,
    else defer. Probability on other_plant counts for neither, so a plant that may be a
    species the model doesn't know needs that much more certainty to be acted on.
    """
    if defer_below is None:
        return ACTIONS[roles[names[detection["category_id"]]]]
    scores = action_scores(detection, names, roles)
    action = max(("keep", "remove"), key=scores.get)
    return action if scores[action] >= defer_below else "defer"


def apply_decisions(results: list[dict[str, Any]], categories: list[dict[str, Any]],
                    roles: dict[str, str], defer_below: float | None) -> list[dict[str, Any]]:
    """Predictions (COCO results) relabelled so that their class gives the action `decide`
    takes: deferred ones become OTHER_CATEGORY, and one kept or removed against its top
    class (rare: e.g. a harmful top class among many neutral ones that sum to more) becomes
    the most probable class with that action. The decision matrix of the relabelled results
    is then what the robot does. Detection matching ignores classes, so it is unaffected."""
    if defer_below is None:
        return results
    names = {c["id"]: c["name"] for c in categories}
    other_id = next((c["id"] for c in categories if c["name"] == OTHER_CATEGORY), None)
    relabelled = []
    for result in results:
        action = decide(result, names, roles, defer_below)
        if action == ACTIONS[roles[names[result["category_id"]]]]:
            relabelled.append(result)
            continue
        if action == "defer":
            if other_id is None:
                raise ValueError(f"defer_below needs an {OTHER_CATEGORY} category")
            category_id = other_id
        else:
            category_id = int(max(
                (c for c in result["class_scores"]
                 if ACTIONS[roles[names[int(c)]]] == action),
                key=result["class_scores"].get))
        relabelled.append({**result, "category_id": category_id})
    return relabelled


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


def level_accuracy(matrix: np.ndarray, names: list[str], taxa: dict[str, dict[str, str]],
                   roles: dict[str, str] | None = None) -> dict[str, Any]:
    """Of the plants matched to a detection in a class confusion matrix (as in
    `decision_matrix`), how many got a class that is right at each level of LEVELS: same
    species, genus, family, order, and role (only with `roles`).

    Missed plants and detections on no plant are left out (AP and the decisions cover
    them), and so is every pair with OTHER_CATEGORY on either side, since it has no taxonomy;
    `other_plant_pairs` counts those. Per level: {"plants", "correct", "accuracy"}.
    """
    *classes, _ = names
    unknown = [name for name in classes if name != OTHER_CATEGORY and name not in taxa]
    if unknown:
        raise ValueError(f"Classes not in the taxa table: {', '.join(unknown)}; taxonomic "
                         "levels need the species-level classes (taxonomy species)")
    levels = [level for level in LEVELS if level != "role" or roles is not None]

    def same(true: str, predicted: str, level: str) -> bool:
        if true == predicted:
            return True
        if level == "species":
            return False
        if level == "role":
            return roles[true] == roles[predicted]
        return bool(taxa[true][level]) and taxa[true][level] == taxa[predicted][level]

    plants, other_pairs = 0, 0
    correct = dict.fromkeys(levels, 0)
    for i, true in enumerate(classes):
        for j, predicted in enumerate(classes):
            count = int(matrix[i, j])
            if not count:
                continue
            if OTHER_CATEGORY in (true, predicted):
                other_pairs += count
                continue
            plants += count
            for level in levels:
                correct[level] += count * same(true, predicted, level)
    result = {level: {"plants": plants, "correct": correct[level],
                      "accuracy": correct[level] / plants if plants else None}
              for level in levels}
    return result | {"other_plant_pairs": other_pairs}
