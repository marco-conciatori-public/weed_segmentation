import csv
import json

import numpy as np
import pytest

from weedseg.data.coco import encode_mask
from weedseg.decisions import Rules, load_rules, removed_up_to
from weedseg.eppo import load_taxa
from weedseg.paths import resolve
from weedseg.row_evaluation import (
    CLASSIFIER,
    CROP,
    EDGE_CROP,
    OTHER,
    WEED,
    Plants,
    combine,
    compare_cases,
    evidence_of,
    fold_masks,
    judge_positions,
    label_of,
    labelled_plants,
    learned_from,
    log_ratio_table,
    logit,
    look_up,
    main_crops,
    position_alone,
    removable,
    rescaled,
    table_cells,
    twostage_plants,
    unseen_plants,
    veto_cases,
    weeds_caught,
    where_errors_are,
    write_report,
)
from weedseg.rows import POSITIONS, RowSettings

CLASSES = ["BEAVA", "ZEAMX", "CHEAL", "MATCH", "other_plant"]
IS_CROP = np.array([True, True, False, False, False])
ROLES = ["crop", "crop", "harmful", "neutral", "other_plant"]
CELLS = {"row_distance_mm": [25, 50, 100, 200], "spacing_within_mm": 100,
         "spacing": [0.25, 0.5, 0.75], "phase": [0.1, 0.2, 0.3], "size": [0.33, 0.5, 0.75, 1.33, 2]}
SETTINGS = {"crops_lost": [0.0, 0.5], "min_row_plants": 3, "on_row_mm": 50, "smooth": 2,
            "largest_ratio": 50, "min_to_learn": 2, "trust": [1, 0.5], "cells": CELLS,
            "learn_from": ["other_sessions", "same_dataset", "other_datasets"],
            "variants": ["rows", "rows+spacing", "size"],
            "veto": {"spacing_above": 0.5, "phase_below": 0.25}, "evidence": ["called_crop"]}


def plants_of(p_crop, labels, sessions=None, datasets=None, harmful=None, plant_scores=None,
              points=None, frames=None):
    """Plants whose probabilities are `p_crop` on BEAVA and the rest on CHEAL."""
    n = len(p_crop)
    p_crop = np.asarray(p_crop, dtype=float)
    probs = np.zeros((n, len(CLASSES)))
    probs[:, 0], probs[:, 2] = p_crop, 1 - p_crop
    sessions = np.array(sessions if sessions is not None else ["a/s1"] * n, dtype=str)
    return Plants(
        probs, IS_CROP, ROLES, Rules("neutral", {}, defer_below=0.9, plant_defer_below=0.5),
        np.ones(n) if plant_scores is None else np.asarray(plant_scores),
        np.zeros((n, 2)) if points is None else np.asarray(points, dtype=float), np.ones(n),
        np.array(datasets if datasets is not None else [s.split("/")[0] for s in sessions.tolist()],
                 dtype=str),
        sessions, np.arange(n) if frames is None else np.asarray(frames),
        np.array(labels, dtype=str), np.zeros(n, bool) if harmful is None else np.asarray(harmful))


def positions_of(n, **values):
    found = np.full(n, np.nan, dtype=POSITIONS)
    for name, value in values.items():
        found[name] = value
    return found


@pytest.mark.parametrize(("annotation", "tag", "label"), [
    ({}, "crop", CROP),
    ({"iscrowd": 1}, "crop", EDGE_CROP),
    ({}, "weed", WEED),
    ({"iscrowd": 1}, "weed", ""),
    ({}, "other", OTHER),
    ({"iscrowd": 1, "hidden_plant": True, "allowed_classes": ["CHEAL"], "source_label": "weed"},
     "ignore", WEED),
    ({"iscrowd": 1, "hidden_plant": True, "allowed_classes": ["CHEAL"],
      "source_label": "partial_weed"}, "ignore", WEED),
    ({"iscrowd": 1, "hidden_plant": True, "allowed_classes": ["POAAN"], "source_label": "Poa"},
     "ignore", ""),
    ({"iscrowd": 1, "hidden_plant": True, "source_label": "weed"}, "ignore", ""),
    ({"iscrowd": 1, "source_label": "Vegetation"}, "ignore", ""),
])
def test_label_of_tells_the_plants_a_score_may_count(annotation, tag, label):
    assert label_of(annotation, tag) == label


def test_scored_counts_only_named_crops_and_weeds():
    plants = plants_of([0.5] * 4, [CROP, WEED, EDGE_CROP, OTHER])
    assert plants.scored.tolist() == [True, True, False, False]


def test_evidence_of_each_source():
    plants = plants_of([0.9, 0.2, 0.6], [WEED, CROP, ""])
    assert evidence_of(plants, "called_crop").tolist() == [True, False, True]
    assert evidence_of(plants, "labelled").tolist() == [False, True, False]
    assert evidence_of(plants, "all").tolist() == [True, True, True]
    with pytest.raises(ValueError, match="come from"):
        evidence_of(plants, "nothing")


def test_a_cell_says_nothing_without_a_real_row():
    positions = positions_of(3, row_distance=[10, 10, 300], strongest_row=[3, 2, 3],
                             spacing=[0.9, 0.9, 0.9], phase=[0.05, 0.05, 0.05], size_ratio=1.0)
    rows = table_cells(positions, "rows", CELLS, 3)
    assert rows.tolist() == [0, -1, 4]


def test_spacing_makes_new_cells_only_on_a_row():
    both = {"strongest_row": [3, 3], "spacing": [0.1, 0.9], "phase": [0.05, 0.4], "size_ratio": 1.0}
    on = table_cells(positions_of(2, row_distance=[10, 10], **both), "rows+spacing", CELLS, 3)
    off = table_cells(positions_of(2, row_distance=[300, 300], **both), "rows+spacing", CELLS, 3)
    assert on[0] != on[1] and off[0] == off[1]


def test_an_unknown_measure_has_its_own_cell():
    positions = positions_of(2, row_distance=[10, 10], strongest_row=[3, 3], spacing=[0.9, np.nan])
    cells = table_cells(positions, "rows+spacing", CELLS, 3)
    assert cells[0] != cells[1]
    with pytest.raises(ValueError, match="tables"):
        table_cells(positions, "colour", CELLS, 3)


def test_log_ratio_table_favours_the_side_a_cell_is_seen_on():
    cells = np.array([1] * 8 + [2] * 2 + [1] * 2 + [2] * 8)
    is_crop = np.array([True] * 10 + [False] * 10)
    table = log_ratio_table(cells, is_crop, smooth=0, largest_ratio=1000)
    assert table[1] == pytest.approx(np.log(0.8 / 0.2))
    assert table[2] == pytest.approx(np.log(0.2 / 0.8))
    assert look_up(table, np.array([2, 7, 1])) == pytest.approx([table[2], 0, table[1]])


def test_log_ratio_table_is_smoothed_and_limited():
    cells = np.array([1] * 100 + [2] * 100)
    is_crop = np.array([True] * 100 + [False] * 100)
    assert log_ratio_table(cells, is_crop, 0.5, 50)[1] < np.log(201 / 1) + 1e-9
    assert log_ratio_table(cells, is_crop, 0.5, 5)[1] == pytest.approx(np.log(5))
    assert log_ratio_table(cells, is_crop, 0.5, 5)[2] == pytest.approx(-np.log(5))


def test_folds_never_learn_from_the_plants_they_judge():
    plants = plants_of([0.5] * 5, [CROP] * 5, sessions=["a/1", "a/1", "a/2", "b/3", "b/4"])
    for fold in ("other_sessions", "same_dataset", "other_datasets"):
        for judged, learned in fold_masks(plants, fold):
            assert not (judged & learned).any()
    masks = {plants.sessions[judged][0]: learned
             for judged, learned in fold_masks(plants, "same_dataset")}
    assert masks["a/1"].tolist() == [False, False, True, False, False]
    assert masks["b/3"].tolist() == [False, False, False, False, True]
    masks = {plants.sessions[judged][0]: learned
             for judged, learned in fold_masks(plants, "other_sessions")}
    assert masks["a/1"].tolist() == [False, False, True, True, True]
    by_dataset = fold_masks(plants, "other_datasets")
    assert len(by_dataset) == 2
    assert by_dataset[0][1].tolist() == [False, False, False, True, True]
    with pytest.raises(ValueError, match="Learning"):
        fold_masks(plants, "nowhere")


def learning_plants(seed=0):
    """Two sessions where a position says something the classifier doesn't."""
    rng = np.random.default_rng(seed)
    n = 400
    labels = np.array([CROP] * (n // 2) + [WEED] * (n // 2))
    # the classifier is a noisy witness, the cell a better one
    p = np.where(labels == CROP, 0.6, 0.4) + rng.normal(0, 0.25, n)
    cells = np.where(rng.random(n) < 0.9, labels == CROP, labels != CROP).astype(np.int64)
    sessions = np.where(np.arange(n) % 2 == 0, "a/1", "a/2")
    return plants_of(p.clip(0.01, 0.99), labels, sessions=sessions), cells


def test_combine_learns_a_position_and_judges_each_part_from_the_other():
    plants, cells = learning_plants()
    base = logit(plants.p_crop)
    parts = [(judged, learned_from(plants, cells, base, learned))
             for judged, learned in fold_masks(plants, "other_sessions")]
    ratios, stack, weights = combine(base, cells, parts, SETTINGS)
    assert ratios[cells == 1].min() > 0 > ratios[cells == 0].max()
    assert weights[1] > 0.5  # the stack trusts a witness that good
    crops = plants.labels == CROP
    right = ((stack > 0) == crops).mean()
    assert right > ((base > 0) == crops).mean() + 0.05


def test_a_part_with_too_little_to_learn_from_stays_as_the_classifier_says():
    plants, cells = learning_plants()
    base = logit(plants.p_crop)
    parts = [(np.ones(len(base), bool), learned_from(plants, cells, base, plants.labels == CROP))]
    ratios, stack, weights = combine(base, cells, parts, SETTINGS)
    assert (ratios == 0).all() and stack == pytest.approx(base) and weights is None


def test_a_plant_without_a_real_row_keeps_the_classifiers_odds():
    plants, cells = learning_plants()
    cells[:10] = -1
    base = logit(plants.p_crop)
    parts = [(judged, learned_from(plants, cells, base, learned))
             for judged, learned in fold_masks(plants, "other_sessions")]
    ratios, _, _ = combine(base, cells, parts, SETTINGS)
    assert (ratios[:10] == 0).all()


def test_rescaled_gives_the_crops_their_share_and_keeps_each_sides_proportions():
    plants = plants_of([0.5], [CROP])
    plants.probs[0] = [0.3, 0.2, 0.25, 0.15, 0.1]
    found = rescaled(plants, np.array([0.8]))
    assert found[0, IS_CROP].sum() == pytest.approx(0.8) and found.sum() == pytest.approx(1)
    assert found[0, 0] / found[0, 1] == pytest.approx(1.5)
    assert found[0, 2] / found[0, 3] == pytest.approx(0.25 / 0.15)
    assert rescaled(plants, plants.p_crop) == pytest.approx(plants.probs)


def test_removable_is_what_the_decisions_say():
    plants = plants_of([0.05, 0.6, 0.05], [WEED, CROP, WEED], plant_scores=[0.9, 0.9, 0.3])
    plants.probs[0] = [0.05, 0, 0.8, 0.05, 0.1]
    names = dict(enumerate(CLASSES))
    roles = dict(zip(CLASSES, ROLES))
    expected = [removed_up_to({"plant_score": float(s), "score": float(s), "category_id": 0,
                               "class_scores": {str(k): float(v) for k, v in enumerate(row)}},
                              names, roles, 0.5)
                for row, s in zip(plants.probs, plants.plant_scores)]
    found = removable(plants, plants.probs)
    assert [None if np.isnan(v) else pytest.approx(v) for v in found] == expected


def test_weeds_caught_loses_at_most_the_crops_allowed():
    p_crop = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.6, 0.3, 0.05])
    crops = np.array([True, True, True, True, False, False, False, False])
    weeds = ~crops
    # losing no crop: the threshold is the lowest crop, 0.2
    assert weeds_caught(p_crop, crops, weeds, 0.0) == pytest.approx(0.5)
    # a quarter of the crops may go, so the threshold rises to 0.7: all weeds are under it
    assert weeds_caught(p_crop, crops, weeds, 0.25) == pytest.approx(1)
    assert weeds_caught(p_crop, crops & False, weeds, 0.0) is None
    assert weeds_caught(p_crop, crops, weeds & False, 0.0) is None


def compare_settings(**changes):
    return {**SETTINGS, **changes}


def row_field(frames=6, sessions=("a/1", "a/2", "b/3", "b/4")):
    """Four sessions of images: a row of 5 crops, a weed off it. The classifier knows them apart
    in `a` and hasn't a clue in `b`."""
    p, labels, names, frame_ids, points = [], [], [], [], []
    frame = 0
    for session in sessions:
        for _ in range(frames):
            frame += 1
            for k in range(5):
                p.append(0.9 if session[0] == "a" else 0.5)
                labels.append(CROP)
                names.append(session)
                frame_ids.append(frame)
                points.append((k * 200.0, 0.0))
            p.append(0.1 if session[0] == "a" else 0.5)
            labels.append(WEED)
            names.append(session)
            frame_ids.append(frame)
            points.append((300.0, 400.0))
    return plants_of(p, labels, sessions=names, points=points, frames=frame_ids)


def test_compare_cases_gives_the_classifier_and_every_way_of_adding_a_position():
    plants = row_field()
    positions, _ = judge_positions(plants, "labelled", RowSettings())
    cases = compare_cases(plants, positions, compare_settings())
    named = {(r["learn_from"], r["case"], r["dataset"]) for r in cases}
    assert ("", CLASSIFIER, "a") in named
    for fold in SETTINGS["learn_from"]:
        for case in ("rows x1", "rows x0.5", "rows, stack", "rows+spacing x1", "size, stack"):
            assert (fold, case, "b") in named
    base = next(r for r in cases if r["case"] == CLASSIFIER and r["dataset"] == "b")
    assert base["auc"] == pytest.approx(0.5) and base["fixed"] == 0 and base["caused"] == 0
    rows = next(r for r in cases if r["learn_from"] == "other_sessions" and r["dataset"] == "b"
                and r["case"] == "rows x1")
    # position sets apart what the classifier can't, and the 0.5 weeds were called crops
    assert rows["auc"] == pytest.approx(1)
    assert rows["caused"] == 0 and rows["fixed"] == rows["weeds"]
    stack = next(r for r in cases if r["learn_from"] == "other_sessions" and r["dataset"] == "b"
                 and r["case"] == "rows, stack")
    assert stack["weight_position"] is not None
    assert rows["weight_position"] is None


def test_the_control_stack_can_only_recalibrate_the_classifier():
    plants = row_field()
    positions, _ = judge_positions(plants, "labelled", RowSettings())
    cases = compare_cases(plants, positions, compare_settings(learn_from=["other_datasets"]))
    assert {r["case"] for r in cases if r["case"].startswith(CLASSIFIER)} == {
        CLASSIFIER, f"{CLASSIFIER}, stack"}  # no plain case: a ratio of 1 is the classifier itself
    control = {r["dataset"]: r for r in cases if r["case"] == f"{CLASSIFIER}, stack"}
    assert control["a"]["weight_position"] == pytest.approx(0)
    # the position sets apart what the classifier can't; without it, a stack can't either
    assert control["b"]["auc"] == pytest.approx(0.5)
    with_rows = next(r for r in cases if r["case"] == "rows, stack" and r["dataset"] == "b")
    assert with_rows["auc"] == pytest.approx(1)


def test_other_datasets_learns_from_the_reference_when_given_one():
    plants = row_field(sessions=("b/3", "b/4"))
    reference = row_field(sessions=("a/1", "a/2"))
    settings = compare_settings(learn_from=["other_datasets"], variants=["rows"])
    positions, _ = judge_positions(plants, "labelled", RowSettings())
    # with itself as the only dataset there is nothing to learn from
    alone = compare_cases(plants, positions, settings)
    assert next(r for r in alone if r["case"] == "rows x1")["auc"] == pytest.approx(0.5)
    other, _ = judge_positions(reference, "labelled", RowSettings())
    helped = compare_cases(plants, positions, settings, (reference, other))
    assert next(r for r in helped if r["case"] == "rows x1")["auc"] == pytest.approx(1)


def test_where_errors_are_tells_what_a_position_could_fix_and_what_it_risks():
    # wrong crop on a row, wrong crop off it, right crop off it, right crop on it;
    # wrong weed off a row, wrong weed on it, right weed on it, right weed off it;
    # and a wrong crop without a real row
    plants = plants_of([0.2, 0.2, 0.9, 0.9, 0.8, 0.8, 0.1, 0.1, 0.2],
                       [CROP] * 4 + [WEED] * 4 + [CROP])
    positions = positions_of(9, row_distance=[10, 90, 90, 10, 90, 10, 10, 90, 10],
                             strongest_row=[3] * 8 + [2])
    found = where_errors_are(plants, positions, SETTINGS)[0]
    assert found["with_real_row"] == pytest.approx(8 / 9)
    assert found["crops"] == 4 and found["crops_wrong"] == 2 and found["crops_wrong_on_row"] == 1
    assert found["weeds"] == 4 and found["weeds_wrong"] == 2 and found["weeds_wrong_off_row"] == 1
    assert found["could_fix"] == 2 and found["at_risk"] == 2


def test_the_brake_stops_removals_on_a_row_only():
    plants = plants_of([0.05, 0.05, 0.05, 0.05], [CROP, CROP, WEED, WEED],
                       harmful=[False, False, True, True])
    positions = positions_of(4, row_distance=[10, 90, 10, 90], strongest_row=[3] * 4,
                             spacing=[1.0, 1.0, 0.2, 1.0], phase=np.nan)
    found = {r["case"]: r for r in veto_cases(plants, positions, SETTINGS)}
    assert found["no veto"]["crops_removed"] == 1 and found["no veto"]["harmful_removed"] == 1
    assert found["on a row"]["crops_removed"] == 0.5 and found["on a row"]["harmful_removed"] == 0.5
    # too close to the next plant to count as well spaced, so this weed is still removed
    assert found["on a row, well spaced"]["crops_removed"] == 0.5
    assert found["on a row, well spaced"]["harmful_removed"] == 1


def box(x, width=30):
    mask = np.zeros((1000, 2000), bool)
    mask[10:40, x:x + width] = True
    return encode_mask(mask)


CATEGORIES = [{"id": 1, "name": "BEAVA", "supercategory": "crop"},
              {"id": 2, "name": "CHEAL", "supercategory": "weed"},
              {"id": 3, "name": "other_plant", "supercategory": "other"},
              {"id": 4, "name": "ignore", "supercategory": "ignore"}]


def annotation(id_, image, x, category, y=10, **more):
    return {"id": id_, "image_id": image, "category_id": category, "bbox": [x, y, 30, 30],
            "area": 900, "iscrowd": 0, **more}


def ground_truth():
    """Two images of a session at 2 mm/px: three crops in line, a named weed off it, a hidden
    weed, an edge crop, and something only known by its family."""
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 2000, "height": 1000, "dataset": "a",
               "group": "s1", "mm_per_px": 2.0} for i in (1, 2)]
    hidden = {"iscrowd": 1, "hidden_plant": True, "allowed_classes": ["CHEAL"], "eppo": "1PLAK"}
    plants = [annotation(1, 1, 100, 1), annotation(2, 1, 300, 1), annotation(3, 1, 500, 1),
              annotation(4, 1, 700, 1), annotation(5, 2, 100, 1), annotation(6, 2, 300, 1),
              annotation(7, 2, 500, 1), annotation(8, 2, 700, 1),
              annotation(9, 1, 400, 2, y=300),
              annotation(10, 1, 200, 4, y=300, source_label="weed", **hidden),
              annotation(11, 1, 600, 1, y=600, iscrowd=1),
              annotation(12, 1, 800, 4, y=300, source_label="Poa", **hidden)]
    for plant in plants:
        plant["segmentation"] = box(plant["bbox"][0])
    return {"images": images, "categories": CATEGORIES, "annotations": plants,
            "info": {"image_dir": "."}}


def test_labelled_plants_are_in_mm_and_leave_out_what_isnt_known():
    plants = labelled_plants(ground_truth())
    assert len(plants.labels) == 11  # not the plant known by its family
    assert plants.points[0] == pytest.approx([(100 + 15) * 2, (10 + 15) * 2])
    assert plants.sizes[0] == 60
    assert plants.sessions[0] == "a/s1" and plants.datasets[0] == "a"
    assert plants.labels.tolist().count(WEED) == 2 and plants.labels.tolist().count(EDGE_CROP) == 1
    assert main_crops(ground_truth()) == {"a/s1": "BEAVA"}


def test_position_alone_tells_crops_from_weeds_per_session_crop():
    gt = ground_truth()
    plants = labelled_plants(gt)
    positions, geometries = judge_positions(plants, "labelled", RowSettings())
    found = position_alone(plants, positions, geometries, main_crops(gt), 50)
    assert len(found) == 1
    row = found[0]
    assert (row["dataset"], row["crop"], row["sessions"]) == ("a", "BEAVA", 1)
    assert row["crops"] == 8 and row["weeds"] == 2
    assert row["crops_on_row"] == 1 and row["weeds_on_row"] == 0
    assert row["row_auc"] == 1 and row["weeds_in_row"] == 0 and row["spacing_auc"] is None


def write_twostage(tmp_path, run="base"):
    gt = ground_truth()
    gt_path = tmp_path / "val.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")

    def detection(image, x, y, crop):
        scores = {"1": crop, "2": 1 - crop}
        return {"image_id": image, "category_id": 1 if crop > 0.5 else 2, "bbox": [x, y, 30, 30],
                "score": 0.95, "plant_score": 0.95, "segmentation": box(x), "class_scores": scores}
    results = [detection(1, x, 10, 0.9) for x in (100, 300, 500, 700)]
    results += [detection(2, x, 10, 0.9) for x in (100, 300, 500, 700)]
    results += [detection(1, 400, 300, 0.1),  # the named weed
                detection(1, 200, 300, 0.8),  # the hidden weed, called a crop
                detection(1, 600, 600, 0.7),  # the edge crop
                detection(1, 1500, 900, 0.2)]  # on no labelled plant
    both = tmp_path / "twostage" / f"det__{run}" / "eval" / "val_last"
    both.mkdir(parents=True)
    (both / "by_dataset.csv").write_text("dataset\nall\n", encoding="utf-8")
    (both / "predictions.json").write_text(json.dumps(results), encoding="utf-8")
    settings = {"score_threshold": 0.5, "iou_threshold": 0.5, "match_by": "box",
                "rules": "configs/rules/example.yaml", "eppo_taxa": "configs/eppo_taxa.csv",
                "max_images": None}
    (both / "evaluation.json").write_text(
        json.dumps({"split_file": str(gt_path), "seed": 0, "settings": settings}), encoding="utf-8")
    return both


def test_twostage_plants_label_detections_including_those_on_ignore_regions(tmp_path):
    plants = twostage_plants(write_twostage(tmp_path), {})
    assert plants.labels.tolist() == [CROP] * 8 + [WEED, WEED, EDGE_CROP, ""]
    assert plants.points[0] == pytest.approx([(100 + 15) * 2, (10 + 15) * 2])
    assert plants.p_crop[0] == pytest.approx(0.9)
    assert plants.harmful[8]  # CHEAL by the rule file; the hidden weed has no species
    assert not plants.harmful[9]


def write_unseen(tmp_path, run="seen_run"):
    folder = tmp_path / "unseen" / run
    folder.mkdir(parents=True)
    ids = np.array([1, 2, 3, 4, 9, 10])
    classes = np.array(["BEAVA", "CHEAL", "other_plant"])
    probs = np.array([[0.8, 0.1, 0.1]] * 4 + [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    np.savez(folder / "val_last.npz", annotation_ids=ids, probs=probs, classes=classes)
    return folder / "val_last.npz"


def rules_and_taxa():
    taxa = load_taxa(resolve("configs/eppo_taxa.csv"))
    return load_rules(resolve("configs/rules/example.yaml"), taxa), taxa


def test_unseen_plants_read_saved_probabilities_with_labels_from_the_ground_truth(tmp_path):
    rules, taxa = rules_and_taxa()
    plants = unseen_plants(write_unseen(tmp_path), ground_truth(), ["a"], rules, taxa)
    assert plants.labels.tolist() == [CROP] * 4 + [WEED, WEED]
    assert plants.is_crop.tolist() == [True, False, False]
    assert plants.p_crop[0] == pytest.approx(0.8)
    assert plants.points[0] == pytest.approx([(100 + 15) * 2, (10 + 15) * 2])
    assert plants.harmful.tolist() == [False] * 4 + [True, False]
    other = unseen_plants(write_unseen(tmp_path, "other"), ground_truth(), ["z"], rules, taxa)
    assert len(other.labels) == 0


def report_settings(tmp_path, runs=None):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "val.json").write_text(json.dumps(ground_truth()), encoding="utf-8")
    return {
        **SETTINGS, "evidence": ["called_crop"], "learn_from": ["other_sessions"],
        "variants": ["rows"], "split": "val", "checkpoint": "last",
        "twostage_dir": str(tmp_path / "twostage"), "rows": {"same_row_mm": 100},
        "runs": {"seeds": [str(tmp_path / "classifier" / "base")]},
        "unseen": {"probabilities": str(tmp_path / "unseen"), "data_dir": str(tmp_path / "data"),
                   "datasets": ["a"], "rules": "configs/rules/example.yaml",
                   "eppo_taxa": "configs/eppo_taxa.csv",
                   "runs": runs or {"new": [str(tmp_path / "classifier" / "new_run")]}}}


def test_write_report_says_which_runs_have_no_probabilities_yet(tmp_path):
    write_twostage(tmp_path)
    text = write_report(report_settings(tmp_path), tmp_path / "out" / "rows_1")
    assert (tmp_path / "out" / "rows_1.md").read_text(encoding="utf-8") == text
    assert "## Position alone" in text and "## Two-stage detections: seeds" in text
    assert "## Not yet scored" in text and "new_run" in text
    assert "Labelled boxes" not in text
    with open(tmp_path / "out" / "rows_1.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["group"] for r in rows} == {"seeds"} and {r["run"] for r in rows} == {"base"}
    assert {r["case"] for r in rows} == {CLASSIFIER, f"{CLASSIFIER}, stack", "rows x1", "rows x0.5",
                                         "rows, stack"}
    for name in ("where", "veto", "alone", "sessions"):
        assert (tmp_path / "out" / f"rows_1_{name}.csv").exists()


def test_write_report_scores_a_run_once_its_probabilities_are_there(tmp_path):
    write_twostage(tmp_path)
    write_unseen(tmp_path, "new_run")
    text = write_report(report_settings(tmp_path), tmp_path / "rows_1")
    assert "## Labelled boxes, new: a" in text and "## Not yet scored" not in text
    with open(tmp_path / "rows_1.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["group"] for r in rows} == {"seeds", "unseen new"}


def test_write_report_says_which_evaluation_is_missing(tmp_path):
    (tmp_path / "twostage").mkdir()
    with pytest.raises(FileNotFoundError, match="predict_twostage"):
        write_report(report_settings(tmp_path), tmp_path / "rows_1")
