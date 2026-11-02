import csv

import numpy as np
import pytest

from weedseg.decisions import decide
from weedseg.unseen import (
    ALL_SITES,
    actions,
    plant_label,
    score_groups,
    score_plants,
    scored_on,
    site_of,
    subset,
    table_rows,
    trained_on,
    write_results,
)

CLASSES = ["BEAVA", "CHEAL", "AMARE", "other_plant"]
ROLES = {"BEAVA": "crop", "CHEAL": "harmful", "AMARE": "harmful", "other_plant": "other_plant"}
TAXA = {"BEAVA": {"genus": "1BEAG"}, "CHEAL": {"genus": "1CHEG"}, "AMARE": {"genus": "1AMAG"},
        "AMATU": {"genus": "1AMAG"}, "ALLPO": {"genus": "1ALLG"}}


def plant(category="BEAVA", allowed=(), dataset="vcd", split="val", site="montoldre"):
    return {"category": category, "eppo": "1PLAK" if allowed else category,
            "allowed": list(allowed), "dataset": dataset, "split": split, "site": site}


def test_trained_on_reads_the_report_or_else_the_folder_name(tmp_path):
    new = tmp_path / "public7" / "species"
    new.mkdir(parents=True)
    (new / "classes.txt").write_text("Datasets: acre, vcd\n3 classes\n", encoding="utf-8")
    assert trained_on(new) == {"acre", "vcd"}
    old = tmp_path / "acre+oppd" / "species"
    old.mkdir(parents=True)
    (old / "classes.txt").write_text("3 classes\n", encoding="utf-8")
    assert trained_on(old) == {"acre", "oppd"}
    assert trained_on(tmp_path / "acre+phenobench" / "species") == {"acre", "phenobench"}


def test_a_run_scores_val_always_and_other_splits_only_of_new_datasets():
    assert scored_on("val", "vcd", {"vcd"})
    assert scored_on("val", "vcd", {"acre"})
    assert scored_on("train", "vcd", {"acre"})
    assert not scored_on("train", "vcd", {"vcd"})


def test_subset_keeps_the_images_of_those_datasets_and_their_annotations():
    coco = {"info": {"image_dir": "."}, "categories": [],
            "images": [{"id": 1, "dataset": "vcd"}, {"id": 2, "dataset": "acre"}],
            "annotations": [{"id": 5, "image_id": 1}, {"id": 6, "image_id": 2}]}
    kept = subset(coco, ["vcd"])
    assert [image["id"] for image in kept["images"]] == [1]
    assert [a["id"] for a in kept["annotations"]] == [5]
    assert kept["info"] == coco["info"] and len(coco["images"]) == 2


def test_site_of_finds_the_site_of_datasets_that_have_sites():
    patterns = {"vcd": "montoldre|lanxade|misc"}
    assert site_of({"dataset": "vcd", "group": "2020-10-01_lanxade_maize_4"}, patterns) == "lanxade"
    assert site_of({"dataset": "vcd", "group": "misc_bean_4"}, patterns) == "misc"
    assert site_of({"dataset": "vcd", "group": "elsewhere"}, patterns) is None
    assert site_of({"dataset": "imageweeds", "group": "1350-1399"}, patterns) is None


@pytest.mark.parametrize("defer_below", [0.5, 0.7, 0.9])
def test_actions_are_those_of_decide(defer_below):
    probs = np.array([[0.95, 0.03, 0.01, 0.01],  # crop, sure
                      [0.02, 0.50, 0.46, 0.02],  # two harmful species: sure of the action
                      [0.45, 0.45, 0.05, 0.05],  # harmful a little ahead, unsure
                      [0.50, 0.25, 0.25, 0.00],  # a tie: keep wins, as in decide
                      [0.05, 0.05, 0.05, 0.85]])  # other_plant counts for neither action
    names = dict(enumerate(CLASSES))
    expected = [decide({"category_id": int(p.argmax()), "score": float(p.max()),
                        "class_scores": {str(i): float(v) for i, v in enumerate(p)}},
                       names, ROLES, defer_below) for p in probs]
    assert actions(probs, CLASSES, ROLES, defer_below).tolist() == expected


def test_score_plants_of_a_species_the_run_has():
    probs = np.array([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0],
                      [0.1, 0.0, 0.1, 0.8], [0.6, 0.4, 0.0, 0.0]])
    scores = score_plants([plant()] * 4, probs, CLASSES, ROLES, TAXA, 0.7, 0.7)
    assert scores["plants"] == 4
    assert scores["right"] == scores["right_genus"] == 0.5
    assert scores["sure"] == 0.75
    assert scores["other_plant"] == 0.25
    assert (scores["keep"], scores["remove"], scores["defer"]) == (0.25, 0.25, 0.5)
    assert scores["most_predicted"] == "BEAVA 50%; CHEAL 25%"
    assert "ruled_out_top" not in scores


def test_score_plants_of_a_species_the_run_lacks_has_no_right_but_may_have_right_genus():
    probs = np.array([[0.1, 0.1, 0.8, 0.0], [0.1, 0.8, 0.1, 0.0]])
    scores = score_plants([plant("AMATU")] * 2, probs, CLASSES, ROLES, TAXA, 0.7, 0.7)
    assert scores["right"] is None
    assert scores["right_genus"] == 0.5
    # A species of no genus the run knows: never right, never by accident of empty codes.
    scores = score_plants([plant("ALLPO")] * 2, probs, CLASSES, ROLES, {**TAXA, "ALLPO": {}},
                          0.7, 0.7)
    assert scores["right_genus"] == 0.0


def test_score_plants_of_a_partial_label_counts_ruled_out_classes_by_probability():
    # Top class BEAVA at 0.3 and at 0.8; a third whose top class is allowed.
    probs = np.array([[0.3, 0.25, 0.25, 0.2], [0.8, 0.1, 0.1, 0.0], [0.1, 0.8, 0.1, 0.0]])
    scores = score_plants([plant("ignore", allowed=["CHEAL", "AMARE", "other_plant"])] * 3, probs,
                          CLASSES, ROLES, TAXA, 0.7, 0.7)
    assert scores["ruled_out_top"] == pytest.approx(2 / 3)
    assert scores["ruled_out_sure"] == pytest.approx(1 / 3)
    assert "right" not in scores


def test_score_groups_splits_by_label_and_site_and_adds_all_sites():
    rows = [plant(site="montoldre"), plant(site="lanxade"), plant("CHEAL", site="lanxade"),
            plant(dataset="imageweeds", site=None)]
    probs = np.array([[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0], [0.1, 0.9, 0, 0], [0.9, 0.1, 0, 0]])
    groups = score_groups(rows, probs, CLASSES, ROLES, TAXA, 0.7, 0.7, CLASSES)
    assert set(groups) == {("vcd", "val", "BEAVA", ALL_SITES), ("vcd", "val", "BEAVA", "montoldre"),
                           ("vcd", "val", "BEAVA", "lanxade"), ("vcd", "val", "CHEAL", ALL_SITES),
                           ("vcd", "val", "CHEAL", "lanxade"),
                           ("imageweeds", "val", "BEAVA", ALL_SITES)}
    assert groups["vcd", "val", "BEAVA", ALL_SITES]["right"] == 0.5
    assert groups["vcd", "val", "BEAVA", "montoldre"]["right"] == 1.0
    assert groups["vcd", "val", "BEAVA", "lanxade"]["right"] == 0.0


def test_plant_label_tells_what_a_partial_label_allows():
    assert plant_label(plant(), CLASSES) == "BEAVA"
    allowed = ["CHEAL", "AMARE", "other_plant"]
    assert plant_label(plant("ignore", allowed=allowed), CLASSES) == "1PLAK: every class but BEAVA"


def make_runs():
    rows = [plant("AMARE", dataset="imageweeds", site=None, split="val")]
    known = score_groups(rows, np.array([[0.1, 0.1, 0.8, 0.0]]), CLASSES, ROLES, TAXA, 0.7, 0.7,
                         CLASSES)
    return [{"name": "new_run", "datasets": {"acre"}, "classes": CLASSES, "scores": known},
            {"name": "seen_run", "datasets": {"imageweeds"}, "classes": CLASSES, "scores": {}}]


def test_table_marks_new_and_seen_and_leaves_groups_a_run_did_not_score_empty():
    table = table_rows(make_runs(), 0.7)
    assert table[0] == ["", "A", "B"]
    assert table[1] == ["**imageweeds val, AMARE, all sites: 1 plants**", "new", "seen"]
    by_name = {row[0]: row[1:] for row in table[1:]}
    assert by_name["right"] == ["100.0%", "-"]
    assert by_name["keep / remove / defer"] == ["0.0% / 100.0% / 0.0%", "-"]
    assert by_name["most predicted"] == ["AMARE 100%", "-"]


def test_write_results_writes_markdown_and_one_csv_row_per_run_and_group(tmp_path):
    text = write_results(make_runs(), 0.7, 0.9, tmp_path / "out" / "unseen_x")
    assert text.startswith("A: new_run (4 classes; trained on acre)")
    assert (tmp_path / "out" / "unseen_x.md").read_text(encoding="utf-8") == text
    with open(tmp_path / "out" / "unseen_x.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(row["letter"], row["seen"], row["plant"]) for row in rows] == [("A", "False", "AMARE")]
