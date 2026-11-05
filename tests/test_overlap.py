import csv
import json
import subprocess
import sys
from dataclasses import replace

import cv2
import numpy as np
import pytest

from weedseg.config import load_config
from weedseg.decisions import Rules
from weedseg.frames import (
    _register_group,
    features,
    frame_groups,
    frame_order,
    frame_shift,
    register,
    registrations,
    usable,
)
from weedseg.overlap import (
    ALL_FRAMES,
    METHODS,
    ONE_FRAME,
    PlantSet,
    combine_views,
    combined,
    corners_iou,
    cut_off,
    evaluate_set,
    frame_summary,
    gain_interval,
    interval_counts,
    intervals,
    label_conflicts,
    link_frames,
    link_summary,
    plant_links,
    project,
    report,
    session_counts,
    variants,
    write_report,
)
from weedseg.paths import resolve
from weedseg.skew_evaluation import ALL, Evaluation, scores
from weedseg.tta import PLAIN, as_saved

CLASSES = ["BEAVA", "CHEAL", "SINAR", "other_plant"]
ROLES = {"BEAVA": "crop", "CHEAL": "harmful", "SINAR": "harmful", "other_plant": "neutral"}
SETTINGS = {"crops_removed": [0.05], "hand_off_below": 0.7, "min_class_plants": 1,
            "datasets": ["acre"], "gaps": [1, 2], "side": 400, "features": 2000, "ratio": 0.75,
            "ransac_px": 4, "min_inliers": 15, "max_scale_change": 0.2, "min_shift": 0.02,
            "min_sessions": 2, "interval_draws": 200}
TRANSLATE = [[1.0, 0.0, -100.0], [0.0, 1.0, 0.0]]


def row_of(name, top=0.8):
    """Probabilities with `name` on top and the rest shared."""
    rest = (1 - top) / (len(CLASSES) - 1)
    return np.array([top if k == name else rest for k in CLASSES])


def evaluation(plants, group="s1"):
    """An Evaluation of plants (dataset, true class or None); probs are given separately."""
    return Evaluation(
        CLASSES, {"BEAVA"}, list(range(1, len(CLASSES) + 1)), ROLES,
        Rules("neutral", {}, defer_below=0.9, plant_defer_below=0.5),
        np.zeros((len(plants), len(CLASSES))), np.full(len(plants), 0.9),
        [(d, group) for d, _ in plants],
        np.array([-1 if t is None else CLASSES.index(t) for _, t in plants]), {},
        list(range(len(plants))), [[0, 0, 5, 5]] * len(plants))


def plant_set(plants, probs, tracks_rows=None, augmented=None, boxes=None, image_ids=None,
              groups=None, kind="detections", rows=None):
    own = evaluation(plants)
    n = len(plants)
    return PlantSet(kind, own, np.array(probs), np.array(probs if augmented is None else augmented),
                    boxes or [[10, 10, 20, 20]] * n, image_ids or list(range(n)),
                    groups or ["s1"] * n, rows)


def texture(seed=0, width=800, height=500):
    """A picture SIFT can place: random blobs, blurred."""
    rng = np.random.default_rng(seed)
    image = (rng.random((height, width)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (0, 0), 2.5)


def as_features(image, settings=SETTINGS, tmp_path=None, name="x.png"):
    path = tmp_path / name
    cv2.imwrite(str(path), image)
    return features(path, settings)


def test_frames_are_ordered_as_taken_with_numbers_as_numbers():
    names = ["a_v_1_10_x.jpg", "a_v_1_9_x.jpg", "a_v_2_1_x.jpg"]
    assert sorted(names, key=frame_order) == ["a_v_1_9_x.jpg", "a_v_1_10_x.jpg", "a_v_2_1_x.jpg"]


def test_frame_groups_keeps_one_dataset_per_session():
    images = [{"id": 1, "dataset": "acre", "group": "g", "file_name": "f10.jpg"},
              {"id": 2, "dataset": "acre", "group": "g", "file_name": "f9.jpg"},
              {"id": 3, "dataset": "acre", "group": "h", "file_name": "f1.jpg"},
              {"id": 4, "dataset": "vcd", "group": "g", "file_name": "f1.jpg"}]
    groups = frame_groups(images, "acre")
    assert [info["id"] for info in groups["g"]] == [2, 1]
    assert [info["id"] for info in groups["h"]] == [3]


def test_register_finds_a_shift_at_full_size(tmp_path):
    image = texture(width=1200, height=800)
    a = as_features(image[:, :1000], tmp_path=tmp_path, name="a.png")
    b = as_features(image[:, 200:1200], tmp_path=tmp_path, name="b.png")
    assert a[2] == pytest.approx(0.4)  # worked on at side 400
    matrix, inliers = register(a, b, SETTINGS)
    assert inliers >= 15
    assert matrix[0, 2] == pytest.approx(-200, abs=4)  # full-size pixels, not side 400's
    assert matrix[1, 2] == pytest.approx(0, abs=4)
    assert matrix[0, 0] == pytest.approx(1, abs=0.02)


def test_register_refuses_two_unrelated_pictures(tmp_path):
    a = as_features(texture(1), tmp_path=tmp_path, name="a.png")
    b = as_features(texture(2), tmp_path=tmp_path, name="b.png")
    result = register(a, b, SETTINGS)
    assert result is None or result[1] < SETTINGS["min_inliers"]


def test_register_copes_with_a_blank_frame(tmp_path):
    blank = as_features(np.full((300, 300), 128, np.uint8), tmp_path=tmp_path, name="blank.png")
    other = as_features(texture(), tmp_path=tmp_path, name="other.png")
    assert register(blank, other, SETTINGS) is None


def test_frames_of_different_sizes_are_registered_in_full_size_pixels(tmp_path):
    image = texture(width=1600, height=1000)
    a = as_features(image[:, :1000], tmp_path=tmp_path, name="a.png")
    half = cv2.resize(image[:, 200:1000], (400, 500), interpolation=cv2.INTER_AREA)
    b = as_features(half, tmp_path=tmp_path, name="b.png")
    matrix, _ = register(a, b, SETTINGS)
    assert matrix[0, 0] == pytest.approx(0.5, abs=0.02)  # b is half the size of a
    assert matrix[0, 2] == pytest.approx(-100, abs=5)


def test_each_frame_is_registered_with_the_neighbours_within_the_gaps(tmp_path):
    image = texture(width=1400, height=500)
    frames = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        cv2.imwrite(str(path), image[:, 100 * i:800 + 100 * i])
        frames.append((i, path))
    found = _register_group((frames, SETTINGS))
    assert {(pair["a"], pair["b"]) for pair in found} == {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}


def test_registrations_are_kept_and_redone_only_for_other_frames_or_settings(tmp_path, monkeypatch):
    calls = []

    def fake(job):
        calls.append(job)
        return [{"a": 1, "b": 2, "matrix": TRANSLATE, "inliers": 50}]

    monkeypatch.setattr("weedseg.frames._register_group", fake)
    groups = {"g": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}]}
    path = tmp_path / "sub" / "val_acre.json"
    found = registrations(groups, tmp_path, SETTINGS, path, 0)
    assert found[0]["inliers"] == 50 and path.exists() and not path.with_suffix(".partial").exists()
    assert registrations(groups, tmp_path, SETTINGS, path, 0) == found
    assert len(calls) == 1
    # What only filters pairs needs no new registration; what places frames does.
    registrations(groups, tmp_path, {**SETTINGS, "min_inliers": 99}, path, 0)
    assert len(calls) == 1
    registrations(groups, tmp_path, {**SETTINGS, "side": 500}, path, 0)
    assert len(calls) == 2
    registrations({"g": groups["g"][:1]}, tmp_path, {**SETTINGS, "side": 500}, path, 0)
    assert len(calls) == 3


@pytest.mark.parametrize(("change", "kept"), [
    ({}, True),
    ({"inliers": 14}, False),  # too few points agree
    ({"matrix": [[1.5, 0, -100], [0, 1.5, 0]]}, False),  # another scale
    ({"matrix": [[1, 0, 0], [0, 1, 0]]}, False),  # marks on the rig: no move
    ({"matrix": [[1, 0, -10], [0, 1, 0]]}, False),  # a robot standing still
])
def test_usable_pairs(change, kept):
    pair = {"matrix": TRANSLATE, "inliers": 50, **change}
    assert usable(pair, (800, 500), SETTINGS) is kept


def test_frame_shift_is_the_move_of_the_centre_so_a_turn_on_the_spot_is_none():
    assert frame_shift(TRANSLATE, (800, 500)) == pytest.approx(0.125)
    # A half turn around the centre of an 800 x 500 frame: the centre stays.
    assert frame_shift([[-1, 0, 800], [0, -1, 500]], (800, 500)) == pytest.approx(0)


def test_project_and_iou():
    assert project([150, 50, 40, 20], TRANSLATE) == (50, 50, 90, 70)
    assert project([0, 0, 10, 10], [[0, -1, 10], [1, 0, 0]]) == (0, 0, 10, 10)  # a quarter turn
    assert corners_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert corners_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert corners_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0


SIZES = {1: (800, 500), 2: (800, 500), 3: (800, 500)}
PAIR = {"a": 1, "b": 2, "matrix": TRANSLATE, "inliers": 50}


def test_boxes_are_linked_after_the_move():
    boxes = {1: [(0, [150, 50, 40, 40]), (1, [400, 300, 40, 40])],
             2: [(2, [50, 50, 40, 40]), (3, [400, 300, 40, 40])]}  # 3 sits where 1 was: not 1
    tracks, links = link_frames(boxes, SIZES, [PAIR], 0.3)
    assert tracks == [[0, 2]]
    assert links[0][:2] == (0, 2) and links[0][2] == pytest.approx(1)


def test_a_plant_cut_by_the_next_frame_is_compared_with_what_that_frame_sees():
    # Moved 100 px left, the box runs off the frame: only x 0-40 of it is visible, as in frame 2.
    boxes = {1: [(0, [40, 50, 100, 40])], 2: [(1, [0, 50, 40, 40])]}
    tracks, _ = link_frames(boxes, SIZES, [PAIR], 0.9)
    assert tracks == [[0, 1]]


def test_a_box_links_once_in_a_frame_and_the_best_match_wins():
    boxes = {1: [(0, [150, 50, 40, 40])],
             2: [(1, [55, 50, 40, 40]), (2, [50, 50, 40, 40])]}
    tracks, _ = link_frames(boxes, SIZES, [PAIR], 0.3)
    assert tracks == [[0, 2]]


def test_links_join_across_frames_into_one_plant():
    boxes = {1: [(0, [250, 50, 40, 40])], 2: [(1, [150, 50, 40, 40])], 3: [(2, [50, 50, 40, 40])]}
    pairs = [PAIR, {"a": 2, "b": 3, "matrix": TRANSLATE, "inliers": 50}]
    tracks, links = link_frames(boxes, SIZES, pairs, 0.3)
    assert tracks == [[0, 1, 2]]
    assert len(links) == 2


def test_nothing_links_under_the_iou_or_outside_the_frame():
    far = {1: [(0, [150, 50, 40, 40])], 2: [(1, [300, 300, 40, 40])]}
    assert link_frames(far, SIZES, [PAIR], 0.3) == ([], [])
    outside = {1: [(0, [10, 50, 40, 40])], 2: [(1, [10, 50, 40, 40])]}  # moves to x -90 to -50
    assert link_frames(outside, SIZES, [PAIR], 0.3) == ([], [])
    assert link_frames({1: [(0, [150, 50, 40, 40])]}, SIZES, [PAIR], 0.3) == ([], [])


def test_cut_off_means_touching_the_frame():
    assert cut_off([0, 50, 10, 10], (100, 100), 3)
    assert cut_off([50, 2, 10, 10], (100, 100), 3)
    assert cut_off([90, 50, 8, 10], (100, 100), 3)
    assert not cut_off([50, 50, 10, 10], (100, 100), 3)


def test_mean_averages():
    probs = np.array([[0.6, 0.4], [0.8, 0.2]])
    assert combine_views(probs, "mean", np.array([True, True])) == pytest.approx([0.7, 0.3])


def test_product_sharpens_views_that_agree_and_ignores_what_the_pipeline_dropped():
    probs = np.array([[0.6, 0.4, 0.0], [0.6, 0.4, 0.0]])
    found = combine_views(probs, "product", np.array([True, True]))
    assert found[0] == pytest.approx(0.36 / 0.52, abs=1e-3)
    assert found[0] > 0.6
    assert found.sum() == pytest.approx(1)
    # A class one view dropped isn't ruled out by that alone.
    mixed = combine_views(np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]), "product",
                          np.array([True, True]))
    assert (mixed > 0).all()


def test_whole_first_trusts_views_not_cut_off():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]])
    assert combine_views(probs, "whole first", np.array([False, True, True])) == pytest.approx(
        [0.3, 0.7])
    # With no whole view, all of them.
    assert combine_views(probs, "whole first", np.zeros(3, bool)) == pytest.approx(probs.mean(0))


def test_an_unknown_way_of_combining_is_refused():
    with pytest.raises(ValueError, match="method"):
        combine_views(np.ones((1, 2)), "median", np.array([True]))


def test_plants_seen_once_stay_exactly_as_they_are():
    plain = np.array([row_of("BEAVA", 0.81234), row_of("CHEAL", 0.6), row_of("BEAVA", 0.9)])
    found = combined(plain, [[1, 2]], "mean", np.ones(3, bool))
    assert np.array_equal(found[0], plain[0])
    assert found[1] == pytest.approx(found[2])
    assert found[1] == pytest.approx(as_saved(plain[[1, 2]].mean(0, keepdims=True))[0])
    assert np.array_equal(combined(plain, [], "mean", np.ones(3, bool)), plain)


def two_views():
    """Three plants: 0 and 1 are one plant, wrong in 1; 2 is seen once and wrong."""
    plants = [("acre", "BEAVA"), ("acre", "BEAVA"), ("acre", "CHEAL")]
    probs = [row_of("BEAVA", 0.9), row_of("CHEAL", 0.4), row_of("BEAVA", 0.8)]
    return plant_set(plants, probs, augmented=[row_of("BEAVA", 0.7), row_of("BEAVA", 0.5),
                                               row_of("CHEAL", 0.9)])


def test_variants_change_only_plants_seen_more_than_once():
    plants = two_views()
    found = variants(plants, [[0, 1]], np.ones(3, bool))
    assert list(found) == [PLAIN, *METHODS, ONE_FRAME, ALL_FRAMES]
    for name, probs in found.items():
        assert np.array_equal(probs[2], plants.plain[2]), name
    assert found[ONE_FRAME][1] == pytest.approx(plants.augmented[1])
    assert found[ALL_FRAMES][0] == pytest.approx(found[ALL_FRAMES][1])


def test_evaluate_set_plain_row_is_the_normal_scores():
    plants = two_views()
    rows = evaluate_set(plants, [[0, 1]], np.ones(3, bool), SETTINGS)
    plain = next(r for r in rows if r["variant"] == PLAIN and r["dataset"] == ALL)
    own = scores(plants.evaluation, plants.plain, SETTINGS)[ALL]
    assert {k: plain[k] for k in own} == own
    assert plain["wrong_to_right"] == plain["right_to_wrong"] == plain["fixes_over_hand_off"] == 0


def test_evaluate_set_counts_fixes_and_who_decides():
    plants = two_views()
    rows = evaluate_set(plants, [[0, 1]], np.ones(3, bool), SETTINGS)
    mean = next(r for r in rows if r["variant"] == "mean" and r["dataset"] == "acre")
    assert mean["seen_twice"] == 2
    assert mean["wrong_to_right"] == 1 and mean["right_to_wrong"] == 0
    # Averaging 0.9 and about 0.2 on BEAVA leaves about 0.55: a fix the hand-off takes away.
    assert mean["fixes_over_hand_off"] == 0
    assert mean["species_right_seen"] == 1
    assert mean["decided"] == pytest.approx(1 / 3)  # only the plant seen once stays over 0.7
    assert mean["right_decided"] == 0  # and that one is wrong
    product = next(r for r in rows if r["variant"] == "product" and r["dataset"] == "acre")
    assert product["wrong_to_right"] == 1


def test_evaluate_set_reports_all_and_the_configured_datasets_only():
    plants = plant_set([("acre", "BEAVA"), ("oppd", "CHEAL")],
                       [row_of("BEAVA"), row_of("CHEAL")])
    rows = evaluate_set(plants, [], np.ones(2, bool), SETTINGS)
    assert {r["dataset"] for r in rows} == {ALL, "acre"}


def test_plant_links_keeps_datasets_apart():
    own = evaluation([("acre", "BEAVA"), ("acre", "BEAVA"), ("vcd", "BEAVA"), ("vcd", "BEAVA")])
    plants = PlantSet("detections", own, np.zeros((4, 4)), np.zeros((4, 4)),
                      [[150, 50, 40, 40], [50, 50, 40, 40], [150, 50, 40, 40], [50, 50, 40, 40]],
                      [1, 2, 1, 2], ["s"] * 4)
    tracks, _ = plant_links(plants, {"acre": [PAIR]}, SIZES, 0.3)
    assert tracks == [[0, 1]]


def test_session_counts_follow_species_right():
    plants = [("acre", "BEAVA"), ("acre", "BEAVA"), ("acre", "other_plant"), ("acre", None),
              ("acre", "CHEAL")]
    own = evaluation(plants)
    probs = np.array([row_of("BEAVA"), row_of("CHEAL"), row_of("BEAVA"), row_of("BEAVA"),
                      row_of("other_plant")])
    counts = session_counts(own, probs, ["b", "a", "a", "a", "b"], np.arange(5))
    # Session a: one plant, wrong (other_plant's label and the unlabelled don't count).
    # Session b: one right; a top class of other_plant doesn't count.
    assert counts.tolist() == [[0, 1], [1, 1]]
    assert session_counts(own, probs, ["a"] * 5, np.array([], int)).shape == (0, 2)


def test_gain_interval_of_no_change_is_nothing_and_a_steady_gain_is_seen():
    before = np.array([[[8.0, 10], [9, 10], [7, 10], [8, 10]]])
    assert gain_interval(before, before, 100) == (0, 0, 0)
    after = before + np.array([[[1.0, 0]]])
    point, low, high = gain_interval(before, after, 200)
    assert point == pytest.approx(0.1)
    assert low == pytest.approx(0.1) and high == pytest.approx(0.1)
    uneven = before.copy()
    uneven[0, 0, 0] += 2
    point, low, high = gain_interval(before, uneven, 500)
    assert point == pytest.approx(0.05)
    assert low <= point <= high and low < high


def test_gain_interval_is_the_mean_of_the_runs():
    before = np.array([[[5.0, 10], [5, 10]], [[5, 10], [5, 10]]])
    after = np.array([[[6.0, 10], [6, 10]], [[8, 10], [8, 10]]])
    assert gain_interval(before, after, 50)[0] == pytest.approx(0.2)


def test_intervals_need_enough_sessions():
    plants = [("acre", "BEAVA")] * 4
    probs = [row_of("BEAVA", 0.9), row_of("CHEAL", 0.4), row_of("BEAVA", 0.9), row_of("BEAVA", 0.9)]
    case = plant_set(plants, probs, groups=["a", "a", "b", "b"])
    tracks = [[0, 1]]
    found = intervals("detections", [interval_counts(case, tracks, np.ones(4, bool), SETTINGS)],
                      SETTINGS)
    mean = next(row for row in found if row["variant"] == "mean")
    assert mean["sessions_all"] == 2 and mean["gain_all"] == pytest.approx(0.25)
    assert mean["sessions_seen"] == 1 and "gain_seen" not in mean  # under min_sessions


def test_link_summary_checks_labels_and_top_classes():
    plants = plant_set(
        [("acre", "BEAVA"), ("acre", "BEAVA"), ("acre", "BEAVA"), ("acre", "CHEAL"),
         ("acre", None), ("acre", "CHEAL")],
        [row_of("BEAVA"), row_of("BEAVA"), row_of("BEAVA"), row_of("CHEAL"), row_of("CHEAL"),
         row_of("CHEAL")])
    links = [(0, 1, 0.9), (2, 3, 0.8), (3, 4, 0.7)]
    row = link_summary(plants, [[0, 1], [2, 3, 4]], links, SETTINGS)[0]
    assert (row["plants"], row["seen_twice"]) == (5, 4)  # the unlabelled one isn't a labelled plant
    assert (row["seen_in_2"], row["seen_in_3_or_more"]) == (2, 2)
    assert (row["links"], row["both_labelled"]) == (3, 2)
    assert row["same_label"] == pytest.approx(0.5)
    assert row["same_top"] == pytest.approx(2 / 3)


def test_label_conflicts_lists_one_plant_named_twice():
    plants = plant_set([("acre", "BEAVA"), ("acre", "CHEAL"), ("acre", "CHEAL"), ("acre", None)],
                       [row_of("BEAVA"), row_of("BEAVA"), row_of("CHEAL"), row_of("CHEAL")],
                       boxes=[[1, 2, 3, 4]] * 4)
    images = {k: {"file_name": f"f{k}.jpg"} for k in range(4)}
    found = label_conflicts(plants, [(0, 1, 0.91234), (1, 2, 0.5), (2, 3, 0.5)], images,
                            [10, 11, 12, 13], "run0")
    assert len(found) == 1
    row = found[0]
    assert (row["label_a"], row["label_b"]) == ("BEAVA", "CHEAL")
    assert (row["called_a"], row["called_b"]) == ("BEAVA", "BEAVA")
    assert (row["annotation_a"], row["annotation_b"]) == (10, 11)
    assert (row["file_a"], row["box_a"]) == ("f0.jpg", "1 2 3 4")
    assert row["iou"] == 0.912


def test_frame_summary_counts_real_neighbours_only():
    groups = {"g": [{"id": 1, "group": "g"}, {"id": 2, "group": "g"}, {"id": 3, "group": "g"}],
              "h": [{"id": 4, "group": "h"}]}
    sizes = dict.fromkeys(range(1, 5), (800, 500))
    kept = [{"a": 1, "b": 2, "matrix": TRANSLATE, "inliers": 50},
            {"a": 1, "b": 3, "matrix": TRANSLATE, "inliers": 50}]
    still = {"a": 2, "b": 3, "matrix": [[1, 0, 0], [0, 1, 0]], "inliers": 900}
    row = frame_summary("acre", groups, [*kept, still], kept, sizes, SETTINGS)
    assert (row["sessions"], row["frames"], row["neighbours"]) == (2, 4, 2)
    assert row["neighbours_overlapping"] == 1
    assert (row["pairs_tried"], row["pairs_matched"], row["pairs_usable"]) == (3, 3, 2)
    assert row["sessions_overlapping"] == 1
    assert row["median_shift"] == pytest.approx(0.125)


def test_report_layout():
    plants = two_views()
    tracks, links = [[0, 1]], [(0, 1, 0.9)]
    whole = np.ones(3, bool)
    results = {
        "scored": [{"run": "r", **row} for row in evaluate_set(plants, tracks, whole, SETTINGS)],
        "links": [{"run": "r", **row} for row in link_summary(plants, tracks, links, SETTINGS)],
        "conflicts": [{"run": "r", "dataset": "acre", "label_a": "BEAVA", "label_b": "CHEAL",
                       "called_a": "BEAVA", "called_b": "BEAVA"}],
        "intervals": intervals("detections",
                               [interval_counts(plants, tracks, whole, SETTINGS)], SETTINGS)}
    frames = [{"dataset": "acre", "sessions": 1, "frames": 3, "neighbours": 2,
               "neighbours_overlapping": 1, "sessions_overlapping": 1, "pairs_matched": 2,
               "pairs_usable": 1, "median_shift": 0.125}]
    text = report(results, frames, SETTINGS, ["r"])
    assert "| acre | 1 | 3 | 1 of 2 | 1 | 2, 1 | 12% |" in text
    assert "# Links" in text and "# Labels across a link" in text
    assert "| acre | BEAVA, CHEAL | 1 | 100% | 0% | 0% |" in text
    assert "## acre: 3 plants, 2 seen more than once" in text
    for name in (PLAIN, *METHODS, ONE_FRAME, ALL_FRAMES):
        assert f"| {name} |" in text
    assert "fixes over hand-off" in text
    assert "# Gain in species right against plain" in text


def test_report_says_when_no_labels_disagree():
    results = {"scored": [], "links": [], "conflicts": [], "intervals": []}
    assert "No link joins two different labels." in report(results, [], SETTINGS, ["r"])


def test_shipped_config_is_usable():
    config = load_config("configs/evaluate_overlap.yaml")
    assert config["split"] == "val"
    assert set(config["datasets"]) == {"acre", "cropandweed", "vcd", "lincolnbeet"}
    assert config["hand_off_below"] == 0.7
    tta = load_config(config["tta_config"])
    assert PLAIN in tta["views"]
    assert json.dumps(config)  # plain values, as kept with the registrations


def test_workers_dont_import_torch():
    # On Windows a worker runs the script's top level and imports its job's module. With torch in
    # there, the laptop ran out of memory with four workers (2026-11-05).
    code = ("import runpy, sys; "
            f"runpy.run_path(r'{resolve('scripts/evaluate_overlap.py')}', run_name='__mp_main__'); "
            "import weedseg.frames; "
            "sys.exit('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_write_report_from_start_to_end(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("data: {dir: d}\n", encoding="utf-8")
    views = tmp_path / "tta" / "run" / "val_last"
    (views / "crops").mkdir(parents=True)
    (views / "crops" / "plants.csv").write_text("kept", encoding="utf-8")
    boxes = [[150, 50, 40, 40], [50, 50, 40, 40], [400, 300, 40, 40]]
    images = [{"id": k, "width": 800, "height": 500, "dataset": "acre", "group": "g",
               "file_name": f"{k}.png"} for k in (1, 2)]
    coco = {"info": {"image_dir": str(tmp_path)}, "images": images,
            "annotations": [{"id": 10 + k, "bbox": box} for k, box in enumerate(boxes)]}
    rows = [{"annotation_id": 10 + k, "image_id": image_id, "category": name, "dataset": "acre",
             "allowed": []} for k, (image_id, name) in enumerate(
                 [(1, "BEAVA"), (2, "CHEAL"), (2, "CHEAL")])]
    probs = np.array([row_of("BEAVA", 0.9), row_of("CHEAL", 0.4), row_of("CHEAL", 0.8)])
    found = replace(evaluation([("acre", "BEAVA"), ("acre", "BEAVA"), ("acre", "CHEAL")]),
                    probs=probs, image_ids=[1, 2, 2], boxes=boxes)
    tta = {"output_dir": str(tmp_path / "tta"), "views": {PLAIN: {}, "flip": {"flip": True}}}
    for name, value in {
            "run_dir": lambda _: run, "load_config": lambda _: tta, "load_coco": lambda _: coco,
            "twostage_eval": lambda *_: tmp_path, "load_evaluation": lambda *_: found,
            "read_index": lambda _: rows, "registrations": lambda *_: [PAIR],
            "collect_views": lambda *_: {PLAIN: probs, "flip": probs}}.items():
        monkeypatch.setattr(f"weedseg.overlap.{name}", value)
    settings = {**SETTINGS, "runs": ["r"], "split": "val", "checkpoint": "last", "min_iou": 0.3,
                "twostage_dir": str(tmp_path), "tta_config": "t", "border_px": 3,
                "num_workers": 0, "output_dir": str(tmp_path / "out")}
    stem = tmp_path / "out" / "overlap_x"
    text = write_report(settings, stem)
    assert stem.with_suffix(".md").read_text(encoding="utf-8") == text
    assert "## acre: 3 plants, 2 seen more than once" in text
    assert "| acre | 1 | 2 | 1 of 1 | 1 | 1, 1 | 12% |" in text
    for suffix in (".csv", "_links.csv", "_intervals.csv"):
        assert stem.with_name(stem.name + suffix).exists()
    # Labelled boxes 0 and 1 are one plant named twice.
    with open(stem.with_name("overlap_x_label_conflicts.csv"), encoding="utf-8") as f:
        conflicts = list(csv.DictReader(f))
    assert [(row["annotation_a"], row["annotation_b"]) for row in conflicts] == [("10", "11")]
    assert "| acre | BEAVA, CHEAL | 1 |" in text
