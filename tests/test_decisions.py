import numpy as np
import pytest

from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import (
    ACTION_COLUMNS,
    TRUE_ROWS,
    apply_decisions,
    decide,
    decision_matrix,
    decision_summary,
    harmful_removed_at,
    level_accuracy,
    load_rules,
    plant_probability,
    removed_up_to,
    role_of,
)


def taxon(code, genus, family, order="1POAO"):
    return {"eppo": code, "genus": genus, "family": family, "order": order}


TAXA = {t["eppo"]: t for t in [
    taxon("ZEAMX", "1ZEAG", "1GRAF"), taxon("SETVI", "1SETG", "1GRAF"),
    taxon("CHEAL", "1CHEG", "1AMAF", "1CAFO"), taxon("MATCH", "1MATG", "1COMF", "1ASTO")]}


def write_rules(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_most_specific_rule_wins(tmp_path):
    rules = load_rules(write_rules(tmp_path / "r.yaml",
                                   "default: neutral\nrules:\n  ZEAMX: crop\n  1GRAF: harmful\n"),
                       TAXA)
    assert role_of("ZEAMX", rules, TAXA) == "crop"  # species beats its family
    assert role_of("SETVI", rules, TAXA) == "harmful"  # via the family
    assert role_of("MATCH", rules, TAXA) == "neutral"  # default
    assert role_of(OTHER_CATEGORY, rules, TAXA) == OTHER_CATEGORY
    with pytest.raises(ValueError, match="not an EPPO code"):
        role_of("grasses", rules, TAXA)


def test_rules_reject_unknown_codes_and_roles(tmp_path):
    with pytest.raises(ValueError, match="1XXXF"):
        load_rules(write_rules(tmp_path / "a.yaml", "default: neutral\nrules:\n  1XXXF: harmful\n"),
                   TAXA)
    with pytest.raises(ValueError, match="weedy"):
        load_rules(write_rules(tmp_path / "b.yaml", "default: weedy\n"), TAXA)


def test_decision_matrix_groups_classes_by_role_and_action():
    names = ["ZEAMX", "SETVI", OTHER_CATEGORY, "background"]
    roles = {"ZEAMX": "crop", "SETVI": "harmful", OTHER_CATEGORY: OTHER_CATEGORY}
    matrix = np.array([[8, 1, 0, 1],  # maize: kept, removed as SETVI, missed
                       [2, 5, 1, 2],  # setaria: kept as maize, removed, deferred, missed
                       [0, 1, 3, 0],  # other_plant
                       [1, 4, 2, 0]])  # detections on no plant
    grouped = decision_matrix(matrix, names, roles)

    def cell(row, column):
        return grouped[TRUE_ROWS.index(row), ACTION_COLUMNS.index(column)]

    assert (cell("crop", "keep"), cell("crop", "remove"), cell("crop", "missed")) == (8, 1, 1)
    assert (cell("harmful", "keep"), cell("harmful", "remove"),
            cell("harmful", "defer")) == (2, 5, 1)
    assert cell("no plant", "remove") == 4
    assert grouped[TRUE_ROWS.index("neutral")].sum() == 0
    summary = decision_summary(grouped)
    assert summary["crops_removed"] == pytest.approx(0.1)
    assert summary["harmful_removed"] == pytest.approx(0.5)
    assert summary["harmful_left"] == pytest.approx(0.4)
    assert summary["neutral_removed"] is None
    assert summary["removals_on_no_plant"] == 4


CATEGORIES = [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "SETVI"},
              {"id": 3, "name": "CHEAL"}, {"id": 4, "name": "MATCH"},
              {"id": 5, "name": OTHER_CATEGORY}]
NAMES = {c["id"]: c["name"] for c in CATEGORIES}
ROLES = {"ZEAMX": "crop", "SETVI": "harmful", "CHEAL": "harmful", "MATCH": "neutral",
         OTHER_CATEGORY: OTHER_CATEGORY}


def result(category_id, score, class_scores=None):
    r = {"category_id": category_id, "score": score}
    return r | {"class_scores": class_scores} if class_scores is not None else r


def two_stage(category_id, plant_score, class_scores):
    return {"category_id": category_id, "score": plant_score, "plant_score": plant_score,
            "class_scores": class_scores}


def test_decide_sums_probabilities_per_action():
    # unsure which harmful weed, sure it is harmful: removed, not deferred
    assert decide(result(2, 0.5, {"2": 0.5, "3": 0.4}), NAMES, ROLES, 0.8) == "remove"
    # crop and neutral both mean keep
    assert decide(result(1, 0.5, {"1": 0.5, "4": 0.35}), NAMES, ROLES, 0.8) == "keep"
    # torn between actions: deferred
    assert decide(result(2, 0.6, {"2": 0.6, "1": 0.35}), NAMES, ROLES, 0.8) == "defer"
    # other_plant counts for no action
    assert decide(result(2, 0.7, {"2": 0.7, "5": 0.25}), NAMES, ROLES, 0.8) == "defer"
    assert decide(result(5, 0.4, {"5": 0.4, "2": 0.3, "3": 0.3}), NAMES, ROLES, 0.5) == "remove"
    # without class scores (older predictions): the top class's score alone
    assert decide(result(2, 0.9), NAMES, ROLES, 0.8) == "remove"
    assert decide(result(2, 0.6), NAMES, ROLES, 0.8) == "defer"
    # without defer_below: the top class's action, other_plant deferred
    assert decide(result(2, 0.3, {"2": 0.3}), NAMES, ROLES, None) == "remove"
    assert decide(result(5, 0.9, {"5": 0.9}), NAMES, ROLES, None) == "defer"


def test_plant_probability_of_each_kind_of_prediction():
    assert plant_probability(result(2, 0.45, {"2": 0.45, "3": 0.4})) == pytest.approx(0.85)
    assert plant_probability(two_stage(2, 0.7, {"2": 0.95, "1": 0.05})) == 0.7
    assert plant_probability(result(2, 0.6)) == 0.6  # older predictions: the score


def test_decide_asks_plant_and_action_separately():
    # species model: sure it is harmful if it is a plant (0.85 / 0.85), plant probability 0.85
    species = result(2, 0.45, {"2": 0.45, "3": 0.4})
    assert decide(species, NAMES, ROLES, 0.9) == "remove"  # no plant threshold
    assert decide(species, NAMES, ROLES, 0.9, 0.8) == "remove"
    assert decide(species, NAMES, ROLES, 0.9, 0.9) == "defer"
    # two-stage: a capped classifier probability no longer drags the plant question down
    capped = two_stage(2, 0.85, {"2": 0.9, "3": 0.05, "1": 0.05})
    assert decide(capped, NAMES, ROLES, 0.9, 0.8) == "remove"
    assert decide(capped, NAMES, ROLES, 0.9, 0.9) == "defer"  # maybe not a plant
    assert decide(capped, NAMES, ROLES, 0.96, 0.8) == "defer"  # unsure what to do
    # the plant threshold alone: the top class's action for plants
    assert decide(capped, NAMES, ROLES, None, 0.8) == "remove"
    assert decide(capped, NAMES, ROLES, None, 0.9) == "defer"
    # older predictions: the score answers both
    assert decide(result(2, 0.85), NAMES, ROLES, 0.8, 0.8) == "remove"
    assert decide(result(2, 0.85), NAMES, ROLES, 0.8, 0.9) == "defer"


def test_removed_up_to_is_where_decide_stops_removing():
    detections = [two_stage(2, 0.95, {"2": 0.6, "3": 0.3, "1": 0.1}),  # remove 0.9
                  two_stage(2, 0.95, {"2": 0.5, "1": 0.2, "5": 0.3}),  # remove 0.5
                  two_stage(2, 0.95, {"2": 0.4, "1": 0.4, "5": 0.2}),  # a tie keeps
                  two_stage(1, 0.95, {"1": 0.9, "2": 0.1}),  # kept
                  two_stage(2, 0.7, {"2": 0.9, "1": 0.1})]  # maybe not a plant
    values = [removed_up_to(d, NAMES, ROLES, 0.8) for d in detections]
    assert values[0] == pytest.approx(0.9) and values[1] == pytest.approx(0.5)
    assert values[2:] == [None, None, None]
    assert removed_up_to(detections[4], NAMES, ROLES) == pytest.approx(0.9)  # no plant gate
    for detection, value in zip(detections, values):
        for defer_below in np.linspace(0, 1, 41):
            removed = decide(detection, NAMES, ROLES, defer_below, 0.8) == "remove"
            assert removed == (value is not None and defer_below <= value)


def test_harmful_removed_at_one_share_of_crops_removed():
    crops, harmful = [0.95, 0.9, None, None], [0.97, 0.92, None]
    assert harmful_removed_at(crops, harmful, 0.0) == (pytest.approx(1 / 3), 0.951)
    assert harmful_removed_at(crops, harmful, 0.25) == (pytest.approx(2 / 3), 0.901)
    assert harmful_removed_at(crops, harmful, 0.5) == (pytest.approx(2 / 3), 0.5)  # lowest
    assert harmful_removed_at(crops, harmful, 0.0, lowest=0.99) == (0.0, 0.99)
    # a 3-decimal probability still gets a threshold above it
    assert harmful_removed_at([0.927], [0.9275, 0.93], 0.0) == (0.5, 0.928)
    assert harmful_removed_at([1.0], harmful, 0.0) == (None, None)  # can't be kept that low
    assert harmful_removed_at(crops, [], 0.0) == (None, 0.951)
    assert harmful_removed_at([], harmful, 0.0) == (pytest.approx(2 / 3), 0.5)


def test_apply_decisions_relabels_to_the_action_taken():
    results = [
        result(2, 0.5, {"2": 0.5, "3": 0.4}),  # removed: unchanged
        result(2, 0.6, {"2": 0.6, "1": 0.35}),  # deferred: other_plant
        # harmful on top, but the neutral and crop classes add up to keep: most probable keeper
        result(3, 0.15, {"3": 0.15, "4": 0.14, "1": 0.7}),
    ]
    decided = apply_decisions(results, CATEGORIES, ROLES, 0.8)
    assert [r["category_id"] for r in decided] == [2, 5, 1]
    assert results[1]["category_id"] == 2  # input untouched
    assert apply_decisions(results, CATEGORIES, ROLES, None) is results
    with pytest.raises(ValueError, match=OTHER_CATEGORY):
        apply_decisions(results, CATEGORIES[:4], ROLES, 0.8)
    # the plant threshold alone also relabels: maybe not a plant -> other_plant
    plants = [two_stage(2, 0.85, {"2": 0.9, "1": 0.1}), two_stage(2, 0.95, {"2": 0.9, "1": 0.1})]
    assert [r["category_id"] for r in apply_decisions(plants, CATEGORIES, ROLES, None, 0.9)] == [
        5, 2]


def test_thresholds_must_be_probabilities(tmp_path):
    for name in ("defer_below", "plant_defer_below"):
        with pytest.raises(ValueError, match=f"{name} .* between 0 and 1"):
            load_rules(write_rules(tmp_path / "bad.yaml", f"default: neutral\n{name}: 80\n"),
                       TAXA)
    rules = load_rules(write_rules(tmp_path / "ok.yaml",
                                   "default: neutral\ndefer_below: 0.9\nplant_defer_below: 0.8\n"),
                       TAXA)
    assert (rules.defer_below, rules.plant_defer_below) == (0.9, 0.8)
    assert load_rules(write_rules(tmp_path / "off.yaml", "default: neutral\n"),
                      TAXA).plant_defer_below is None


def test_level_accuracy_counts_found_plants_right_at_each_level():
    names = ["ZEAMX", "SETVI", "CHEAL", OTHER_CATEGORY, "background"]
    roles = {"ZEAMX": "crop", "SETVI": "harmful", "CHEAL": "harmful",
             OTHER_CATEGORY: OTHER_CATEGORY}
    matrix = np.array([[6, 2, 0, 1, 3],  # maize: right, as SETVI (same family), as other
                       [0, 4, 1, 0, 2],  # setaria: right, as CHEAL (same role only)
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 2, 0],  # other_plant pairs are left out
                       [5, 0, 0, 0, 0]])  # detections on no plant are left out
    levels = level_accuracy(matrix, names, TAXA, roles)
    assert levels["other_plant_pairs"] == 4
    assert {k: v["correct"] for k, v in levels.items() if k != "other_plant_pairs"} == {
        "species": 10, "genus": 10, "family": 12, "order": 12, "role": 11}
    assert levels["species"]["plants"] == 13
    assert levels["family"]["accuracy"] == pytest.approx(12 / 13)
    assert "role" not in level_accuracy(matrix, names, TAXA)
    with pytest.raises(ValueError, match="weed"):
        level_accuracy(np.zeros((3, 3)), ["crop", "weed", "background"], TAXA)
