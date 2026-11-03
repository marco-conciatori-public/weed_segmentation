import csv
import json

import numpy as np
import pytest

from weedseg.data.coco import encode_mask
from weedseg.decisions import Rules
from weedseg.skew_evaluation import (
    ALL,
    Evaluation,
    Lists,
    first_pass_lists,
    labelled_lists,
    load_evaluation,
    missed_weeds,
    scores,
    skewed,
    write_report,
)

CLASSES = ["BEAVA", "ZEAMX", "CHEAL", "ECHCG", "MATCH", "other_plant"]
ROLES = {"BEAVA": "crop", "ZEAMX": "crop", "CHEAL": "harmful", "ECHCG": "harmful",
         "MATCH": "neutral", "other_plant": "other_plant"}
SETTINGS = {"crops_removed": [0.0, 0.5], "hand_off_below": 0.7, "min_class_plants": 1}


def row_of(name, top=0.7):
    """Probabilities with `name` on top and the rest shared out."""
    probs = np.full(len(CLASSES), (1 - top) / (len(CLASSES) - 1))
    probs[CLASSES.index(name)] = top
    return probs


def evaluation(plants, top=0.7, labelled=None):
    """An Evaluation of plants (session, predicted class, true class or None)."""
    return Evaluation(
        CLASSES, {"BEAVA", "ZEAMX"}, list(range(1, len(CLASSES) + 1)), ROLES,
        Rules("neutral", {}, defer_below=0.9, plant_defer_below=0.5),
        np.array([row_of(p, top) for _, p, _ in plants]), np.full(len(plants), 0.9),
        [s for s, _, _ in plants],
        np.array([-1 if t is None else CLASSES.index(t) for _, _, t in plants]), labelled or {})


def session_of(counts, session=("a", "s1")):
    return [(session, name, name) for name, n in counts.items() for _ in range(n)]


def test_labelled_lists_split_crops_from_weeds_and_leave_out_other_plant():
    found = evaluation([], labelled={("a", "s1"): {"BEAVA", "CHEAL", "other_plant"},
                                     ("a", "s2"): {"other_plant"}})
    lists = labelled_lists(found)
    assert lists["a", "s1"] == Lists(frozenset({"BEAVA"}), frozenset({"CHEAL"}))
    assert lists["a", "s2"] == Lists()


def test_a_first_pass_lists_the_top_crop_and_weeds_by_their_share_of_weeds():
    plants = session_of({"BEAVA": 28, "ZEAMX": 2, "CHEAL": 7, "ECHCG": 1, "other_plant": 2})
    found = first_pass_lists(evaluation(plants), 0.1, 40)[("a", "s1")]
    # ECHCG is 1 of 8 weeds: the crops and other_plant don't dilute its share, nor is it listed
    # because of them
    assert found == Lists(frozenset({"BEAVA"}), frozenset({"CHEAL", "ECHCG"}))
    assert first_pass_lists(evaluation(plants), 0.2, 40)[("a", "s1")].weed == {"CHEAL"}


def test_a_first_pass_has_no_list_under_the_minimum_and_no_crop_when_none_is_predicted():
    assert first_pass_lists(evaluation(session_of({"CHEAL": 39})), 0.1, 40) == {}
    found = first_pass_lists(evaluation(session_of({"CHEAL": 40})), 0.1, 40)[("a", "s1")]
    assert found == Lists(frozenset(), frozenset({"CHEAL"}))


def test_small_sessions_are_joined_two_by_two_with_the_same_top_crop_largest_first():
    plants = [*session_of({"BEAVA": 15, "CHEAL": 10}, ("a", "s1")),
              *session_of({"BEAVA": 15, "ECHCG": 5}, ("a", "s2")),
              *session_of({"BEAVA": 5}, ("a", "s3")),  # left over, and too small alone
              *session_of({"ZEAMX": 39}, ("a", "s4")),  # another crop: not joined with s3
              *session_of({"BEAVA": 39}, ("b", "s5")),  # another dataset: not joined either
              *session_of({"ZEAMX": 40}, ("a", "s6"))]  # large enough alone
    found = evaluation(plants)
    assert set(first_pass_lists(found, 0.1, 40)) == {("a", "s6")}
    lists = first_pass_lists(found, 0.1, 40, pair_small=True)
    assert set(lists) == {("a", "s1"), ("a", "s2"), ("a", "s6")}
    weeds = frozenset({"CHEAL", "ECHCG"})
    assert lists["a", "s1"] == Lists(frozenset({"BEAVA"}), weeds, "s2")
    assert lists["a", "s2"] == Lists(frozenset({"BEAVA"}), weeds, "s1")
    assert lists["a", "s6"].partner == ""


def test_a_pair_still_under_the_minimum_gets_no_list():
    plants = [*session_of({"BEAVA": 10}, ("a", "s1")), *session_of({"BEAVA": 10}, ("a", "s2"))]
    assert first_pass_lists(evaluation(plants), 0.1, 40, pair_small=True) == {}


def test_skewed_changes_only_sessions_with_lists_and_only_the_chosen_side():
    plants = [(("a", "s1"), "ZEAMX", "ZEAMX"), (("a", "s1"), "CHEAL", "CHEAL"),
              (("a", "s2"), "CHEAL", "CHEAL")]
    found = evaluation(plants)
    lists = {("a", "s1"): Lists(frozenset({"BEAVA"}), frozenset({"ECHCG"}))}
    both = skewed(found, lists, "both", 5)
    assert both[2] == pytest.approx(found.probs[2])
    beava, zeamx, cheal, echcg = (CLASSES.index(n) for n in ("BEAVA", "ZEAMX", "CHEAL", "ECHCG"))
    assert both[0, zeamx] / both[0, beava] == pytest.approx(
        found.probs[0, zeamx] / found.probs[0, beava] / 5)
    assert both[1, cheal] / both[1, echcg] == pytest.approx(
        found.probs[1, cheal] / found.probs[1, echcg] / 5)
    crop_only, weed_only = skewed(found, lists, "crop", 5), skewed(found, lists, "weed", 5)
    assert crop_only[1, cheal] / crop_only[1, echcg] == pytest.approx(
        found.probs[1, cheal] / found.probs[1, echcg])
    assert weed_only[0, zeamx] / weed_only[0, beava] == pytest.approx(
        found.probs[0, zeamx] / found.probs[0, beava])
    assert skewed(found, {("b", "elsewhere"): lists["a", "s1"]}, "both", 5) == pytest.approx(
        found.probs)


def test_scores_per_dataset_and_overall():
    plants = [(("a", "s"), "BEAVA", "BEAVA"), (("a", "s"), "CHEAL", "BEAVA"),
              (("a", "s"), "CHEAL", "CHEAL"), (("a", "s"), "ZEAMX", "ECHCG"),
              (("a", "s"), "other_plant", "MATCH"), (("a", "s"), "CHEAL", None),
              (("b", "t"), "CHEAL", "CHEAL")]
    found = scores(evaluation(plants, top=0.95), evaluation(plants, top=0.95).probs, SETTINGS)
    a = found["a"]
    assert a["plants"] == 5  # a detection on no labelled plant doesn't count
    assert a["species_right"] == pytest.approx(2 / 4)  # without the plant called other_plant
    assert a["class_right"] == pytest.approx(2 / 5)
    assert a["balanced"] == pytest.approx((1 / 2 + 1 + 0 + 0) / 4)
    assert a["crop_right"] == pytest.approx(3 / 5)  # the beet called CHEAL, ECHCG called maize
    assert a["weeds_called_crop"] == pytest.approx(1 / 3)
    assert a["crops_removed"] == pytest.approx(1 / 2)
    assert a["harmful_removed"] == pytest.approx(1 / 2)  # CHEAL yes, ECHCG (called maize) no
    assert a["under_hand_off"] == 0
    # keeping crops removed at 0: everything the beet's removal allows must go too
    assert a["harmful_at_0.5"] == pytest.approx(1 / 2) and a["defer_below_at_0.5"] == 0.5
    assert found[ALL]["plants"] == 6 and found["b"]["species_right"] == 1
    assert scores(evaluation(plants, top=0.6), evaluation(plants, top=0.6).probs,
                  SETTINGS)["a"]["under_hand_off"] == 1


def test_missed_weeds_leaves_each_weed_off_in_turn():
    plants = [*[(("a", "s1"), "CHEAL", "CHEAL") for _ in range(3)],
              *[(("a", "s1"), "ECHCG", "ECHCG") for _ in range(2)],
              (("a", "s1"), "MATCH", "MATCH"),  # labelled but not on the list
              *[(("a", "s2"), "MATCH", "MATCH") for _ in range(3)]]  # the only weed: nothing left
    found = evaluation(plants, top=0.5)
    lists = {("a", "s1"): Lists(frozenset({"BEAVA"}), frozenset({"CHEAL", "ECHCG"})),
             ("a", "s2"): Lists(frozenset(), frozenset({"MATCH"}))}
    # CHEAL at 0.5 / 1000 falls below ECHCG at 0.1; ECHCG too with CHEAL listed
    missed = missed_weeds(found, lists, 1000, min_plants=2)["a"]
    assert missed["species"] == 2
    assert missed["right_unskewed"] == 1 and missed["right"] == 0
    assert missed["crop_unskewed"] == 0 and missed["crop"] == 0
    # at a low multiplier they stay on top
    assert missed_weeds(found, lists, 2, min_plants=2)["a"]["right"] == 1
    assert missed_weeds(found, lists, 2, min_plants=3)["a"]["species"] == 1  # ECHCG has too few


def box(x):
    mask = np.zeros((100, 200), bool)
    mask[10:40, x:x + 30] = True
    return encode_mask(mask)


def make_evaluation(tmp_path):
    """Two sessions in two datasets; a classifier run with a two-stage evaluation of them."""
    categories = [{"id": 1, "name": "ZEAMX", "supercategory": "crop"},
                  {"id": 2, "name": "CHEAL", "supercategory": "weed"},
                  {"id": 3, "name": "ECHCG", "supercategory": "weed"},
                  {"id": 4, "name": "other_plant", "supercategory": "other"},
                  {"id": 5, "name": "ignore", "supercategory": "ignore"}]
    plants = [(1, 1, 10, 1), (2, 1, 60, 2), (3, 1, 110, 3), (4, 2, 10, 2)]
    gt = {"images": [{"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100, "dataset": d,
                      "group": g} for i, d, g in [(1, "a", "s1"), (2, "b", "s2")]],
          "categories": categories,
          "annotations": [{"id": id_, "image_id": image, "category_id": category,
                           "bbox": [x, 10, 30, 30], "area": 900, "iscrowd": 0,
                           "segmentation": box(x)} for id_, image, x, category in plants]}
    gt_path = tmp_path / "val.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    # the ECHCG plant looks a bit more like CHEAL, which a weed list of ECHCG can put right
    scores_of = [{"1": 0.95, "2": 0.05}, {"2": 0.95, "1": 0.05}, {"2": 0.6, "3": 0.35, "4": 0.05},
                 {"2": 0.95, "4": 0.05}]
    results = [{"image_id": image, "category_id": int(max(s, key=s.get)),
                "bbox": [x, 10, 30, 30], "score": 0.95, "plant_score": 0.95,
                "segmentation": box(x), "class_scores": s}
               for (_, image, x, _), s in zip(plants, scores_of)]
    results.append({**results[0], "score": 0.3, "plant_score": 0.3})  # under the score threshold
    both = tmp_path / "twostage" / "det__base" / "eval" / "val_last"
    both.mkdir(parents=True)
    (both / "by_dataset.csv").write_text("dataset\nall\n", encoding="utf-8")
    (both / "predictions.json").write_text(json.dumps(results), encoding="utf-8")
    settings = {"score_threshold": 0.5, "iou_threshold": 0.5, "match_by": "mask",
                "rules": "configs/rules/example.yaml", "eppo_taxa": "configs/eppo_taxa.csv",
                "max_images": None}
    (both / "evaluation.json").write_text(
        json.dumps({"split_file": str(gt_path), "seed": 0, "settings": settings}),
        encoding="utf-8")
    return both


def report_settings(tmp_path, sources=("labelled", "first_pass", "first_pass_paired")):
    return {"twostage_dir": str(tmp_path / "twostage"), "split": "val", "checkpoint": "last",
            "multipliers": [5], "lists": ["both", "crop", "weed"], "sources": list(sources),
            "first_pass": {"weed_share": 0.1, "min_plants": 2}, "crops_removed": [0.01],
            "hand_off_below": 0.7, "min_class_plants": 1, "missed_min_plants": 1}


def test_load_evaluation_reads_detections_matches_and_labels(tmp_path):
    found = load_evaluation(make_evaluation(tmp_path), {})
    assert found.classes == ["ZEAMX", "CHEAL", "ECHCG", "other_plant"]
    assert found.crops == {"ZEAMX"} and found.ids == [1, 2, 3, 4]
    assert len(found.true) == 4  # the detection under the score threshold is left out
    assert found.true.tolist() == [0, 1, 2, 1]
    assert found.sessions == [("a", "s1")] * 3 + [("b", "s2")]
    assert found.labelled["a", "s1"] == {"ZEAMX", "CHEAL", "ECHCG"}
    assert found.roles["ZEAMX"] == "crop" and found.roles["CHEAL"] == "harmful"
    assert found.probs.sum(1) == pytest.approx(1)
    assert found.rows["a", "s1"].tolist() == [0, 1, 2]


def test_write_report_shows_a_list_putting_a_plant_right(tmp_path):
    run = tmp_path / "classifier" / "base"
    make_evaluation(tmp_path)
    settings = report_settings(tmp_path)
    text = write_report([run], settings, tmp_path / "out" / "skew_1")
    assert (tmp_path / "out" / "skew_1.md").read_text(encoding="utf-8") == text
    assert text.index("## all") < text.index("## a\n") < text.index("## b\n")
    with open(tmp_path / "out" / "skew_1.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cell = {(r["dataset"], r["source"], r["lists"]): r for r in rows}
    assert float(cell["a", "no skew", ""]["species_right"]) == pytest.approx(2 / 3)
    # with the session's labelled weeds nothing changes: CHEAL and ECHCG are both on the list
    assert float(cell["a", "labelled", "both"]["species_right"]) == pytest.approx(2 / 3)
    assert cell["a", "labelled", "both"]["sessions_with_list"] == "1"
    assert cell["b", "labelled", "crop"]["sessions_with_list"] == "0"  # no crop labelled there
    # the model calls only CHEAL in its first pass, so ECHCG is not listed
    assert float(cell["a", "first_pass", "weed"]["species_right"]) == pytest.approx(2 / 3)
    with open(tmp_path / "out" / "skew_1_lists.csv", encoding="utf-8", newline="") as f:
        lists = {(r["source"], r["session"]): r for r in csv.DictReader(f)}
    assert lists["labelled", "s1"]["crop_list"] == "ZEAMX"
    assert lists["labelled", "s1"]["weed_list"] == "CHEAL ECHCG"
    assert lists["labelled", "s1"]["plants"] == "3"
    assert lists["first_pass", "s1"]["weed_list"] == "CHEAL"
    assert ("first_pass", "s2") not in lists  # a single plant is under min_plants
    assert (tmp_path / "out" / "skew_1_missed.csv").exists()


def test_write_report_says_which_evaluation_is_missing(tmp_path):
    (tmp_path / "twostage").mkdir()
    with pytest.raises(FileNotFoundError, match="predict_twostage"):
        write_report([tmp_path / "classifier" / "base"], report_settings(tmp_path),
                     tmp_path / "skew_1")
