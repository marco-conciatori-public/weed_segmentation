import numpy as np
import pytest

from weedseg.data.coco import OTHER_CATEGORY
from weedseg.decisions import (
    ACTION_COLUMNS,
    TRUE_ROWS,
    decision_matrix,
    decision_summary,
    defer_uncertain,
    load_rules,
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


def test_uncertain_detections_are_deferred(tmp_path):
    rules = load_rules(write_rules(tmp_path / "r.yaml", "default: neutral\ndefer_below: 0.8\n"),
                       TAXA)
    categories = [{"id": 1, "name": "SETVI"}, {"id": 2, "name": OTHER_CATEGORY}]
    results = [{"category_id": 1, "score": 0.9}, {"category_id": 1, "score": 0.6}]
    deferred = defer_uncertain(results, categories, rules.defer_below)
    assert [r["category_id"] for r in deferred] == [1, 2]
    assert results[1]["category_id"] == 1  # input untouched
    assert defer_uncertain(results, categories, None) is results
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_rules(write_rules(tmp_path / "bad.yaml", "default: neutral\ndefer_below: 80\n"),
                   TAXA)
