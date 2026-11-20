import numpy as np

from weedseg.decisions import Rules
from weedseg.size_evaluation import (
    CLASSIFIER,
    Plants,
    Stack,
    case_name,
    fit_stack,
    fold_masks,
    score_cases,
    session_parts,
    stacked,
    with_error,
)
from weedseg.skew_evaluation import ALL, Evaluation

CLASSES = ["BEAVA", "CHEAL", "ECHCG", "other_plant"]
ROLES = {"BEAVA": "crop", "CHEAL": "harmful", "ECHCG": "harmful", "other_plant": "other_plant"}
SETTINGS = {"folds": ["other_sessions", "other_datasets"], "session_parts": 2, "l2": 1e-4,
            "min_fit_plants": 5, "max_iter": 100, "gsd_errors": [0.2], "seed": 0,
            "crops_removed": [0.5], "hand_off_below": 0.7, "min_class_plants": 1}


def plants_of(sessions, true, sizes, probs=None):
    """Plants whose classifier can't tell CHEAL from ECHCG (0.45 each), unless `probs`."""
    count = len(true)
    if probs is None:
        probs = np.tile([0.05, 0.45, 0.45, 0.05], (count, 1))
    evaluation = Evaluation(
        CLASSES, {"BEAVA"}, list(range(1, len(CLASSES) + 1)), ROLES,
        Rules("neutral", {}, defer_below=0.9, plant_defer_below=0.5), probs, np.ones(count),
        sessions, np.array([CLASSES.index(t) for t in true]), {}, list(range(count)), [])
    return Plants(evaluation, probs.astype(np.float32), np.array(sizes, dtype=np.float64),
                  np.full(count, 0.5), np.arange(count))


def size_coded(count=200, seed=0):
    """CHEAL about 20 mm, ECHCG about 80 mm, in two datasets of four sessions each."""
    rng = np.random.default_rng(seed)
    true = ["CHEAL" if k % 2 else "ECHCG" for k in range(count)]
    sizes = [(20 if t == "CHEAL" else 80) * np.exp(0.15 * rng.standard_normal()) for t in true]
    sessions = [(f"d{k % 2}", f"s{k % 4}") for k in range(count)]
    return plants_of(sessions, true, sizes)


def test_session_parts_keep_sessions_whole_and_balance_plants():
    sessions = [("a", "1")] * 6 + [("a", "2")] * 3 + [("b", "1")] * 3 + [("b", "2")] * 2
    part = session_parts(sessions, 2)
    for session in set(sessions):
        assert len({p for p, s in zip(part, sessions) if s == session}) == 1
    assert sorted(np.bincount(part).tolist()) == [6, 8]


def test_folds_never_judge_with_what_they_learned_from():
    plants = size_coded()
    for fold in ("other_sessions", "other_datasets"):
        masks = fold_masks(plants, fold, 2)
        judged = np.sum([j for j, _ in masks], 0)
        assert (judged == 1).all()
        for j, learned in masks:
            assert not (j & learned).any()
    datasets = fold_masks(plants, "other_datasets", 2)
    assert len(datasets) == 2


def test_with_error_moves_every_plant_of_an_image_alike():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    found = with_error(values, np.array([7, 7, 9, 9]), 0.3, np.random.default_rng(0))
    ratios = found / values
    assert ratios[0] == ratios[1] and ratios[2] == ratios[3] and ratios[0] != ratios[2]
    assert np.allclose(with_error(values, np.arange(4), 0.0, np.random.default_rng(0)), values)


def test_a_size_stack_learns_what_the_classifier_cannot_see():
    plants = size_coded()
    true = plants.evaluation.true
    log_probs = np.log(plants.probs.astype(np.float64))
    control = fit_stack(log_probs, true, None, SETTINGS)
    size = fit_stack(log_probs, true, plants.size_mm, SETTINGS)
    assert (size.probs(log_probs, plants.size_mm).argmax(1) == true).mean() > 0.95
    assert (control.probs(log_probs, None).argmax(1) == true).mean() <= 0.55
    # BEAVA and other_plant have no plants, so no bias or size term
    assert size.bias[[0, 3]].tolist() == [0, 0] and (size.terms[:, [0, 3]] == 0).all()


def test_unknown_values_change_nothing():
    stack = Stack(1.0, np.zeros(2), np.array([[3.0, -3.0], [1.0, 1.0]]), mean=np.log(10), std=1)
    log_probs = np.log(np.array([[0.7, 0.3]]))
    assert np.allclose(stack.probs(log_probs, np.array([np.nan])), [[0.7, 0.3]])
    assert np.allclose(stack.probs(log_probs, np.array([10.0])), [[0.7, 0.3]])


def test_stacked_judges_each_plant_once_with_another_fold():
    plants = size_coded()
    exact, wrong = stacked(plants, "other_sessions", SETTINGS, plants.size_mm,
                           [plants.size_mm, plants.size_mm * 4])
    true = plants.evaluation.true
    assert (exact.argmax(1) == true).mean() > 0.95
    assert (wrong.argmax(1) == true).mean() < 0.6  # every plant looks four times its size
    assert np.allclose(exact.sum(1), 1, atol=1e-5)


def test_score_cases_covers_every_case_and_the_classifier_is_unchanged():
    rows = score_cases(size_coded(), SETTINGS)
    names = list(dict.fromkeys(case_name(row) for row in rows))
    # classifier, then per fold: control, camera, size, size on wrong, size learned with error
    assert len(names) == 1 + 2 * 5 and names[0] == CLASSIFIER
    first = next(row for row in rows if row["variant"] == CLASSIFIER and row["dataset"] == ALL)
    assert first["wrong_to_right"] == first["right_to_wrong"] == 0
    size = next(row for row in rows if row["variant"] == "size" and row["fold"] == "other_sessions"
                and row["error"] is None and row["dataset"] == ALL)
    control = next(row for row in rows if row["variant"] == "control"
                   and row["fold"] == "other_sessions" and row["dataset"] == ALL)
    assert size["class_right"] > 0.95 > control["class_right"]
    assert size["log_loss"] < control["log_loss"]
