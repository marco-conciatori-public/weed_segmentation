import csv

import cv2
import numpy as np
import pytest
import torch

from weedseg.classifier.crops import cache_dir, read_index, write_cache
from weedseg.classifier.data import CropDataset, view_crop
from weedseg.classifier.model import build_classifier
from weedseg.config import load_config
from weedseg.decisions import Rules
from weedseg.skew_evaluation import ALL, Evaluation, scores, write_csv
from weedseg.tta import (
    PLAIN,
    Variant,
    aggregate,
    as_saved,
    check_scales,
    check_views,
    collect_views,
    combine,
    crop_evaluation,
    crop_rows,
    crop_view,
    detection_view,
    evaluate_variants,
    gated,
    partial_fit,
    pick,
    plain_checks,
    report,
    save_view,
    saved_view,
    subset_evaluation,
    variants,
)
from weedseg.twostage import classify_detections, detection_input, with_species

CLASSES = ["BEAVA", "CHEAL", "ECHCG", "other_plant"]
ROLES = {"BEAVA": "crop", "CHEAL": "harmful", "ECHCG": "harmful", "other_plant": "other_plant"}
CROPS = {"margin": 0.25, "input_size": 28, "cache_context": 1.5, "cache_quality": 95,
         "background": "keep", "dim_factor": 0.3}
TINY_MODEL = {"backbone": "vit_small_patch14_dinov2.lvd142m", "pretrained": False,
              "pretrained_path": None, "freeze_blocks": 0, "drop_path": 0.0}
VIEWS = {PLAIN: {}, "flip": {"flip": True}, "looser": {"scale": 1.15}}
SETTINGS = {"views": VIEWS, "view_sets": {"two": [PLAIN, "flip"], "three": [PLAIN, "flip", "looser"]},
            "main_set": "three", "votes": True, "gates": [0.7], "crops_removed": [0.5],
            "hand_off_below": 0.7, "min_class_plants": 1}


def row_of(name, top=0.7):
    """Probabilities with `name` on top and the rest shared out."""
    probs = np.full(len(CLASSES), (1 - top) / (len(CLASSES) - 1))
    probs[CLASSES.index(name)] = top
    return probs


def evaluation(plants, top=0.7):
    """An Evaluation of plants (dataset, predicted class, true class or None)."""
    return Evaluation(
        CLASSES, {"BEAVA"}, list(range(1, len(CLASSES) + 1)), ROLES,
        Rules("neutral", {}, defer_below=0.9, plant_defer_below=0.5),
        np.array([row_of(p, top) for _, p, _ in plants]), np.full(len(plants), 0.9),
        [(d, "s1") for d, _, _ in plants],
        np.array([-1 if t is None else CLASSES.index(t) for _, _, t in plants]), {},
        list(range(len(plants))), [[0, 0, 5, 5]] * len(plants))


def test_view_crop_gives_all_eight_orientations_and_leaves_the_plain_crop_alone():
    crop = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = crop[..., 0] > 20
    assert view_crop(crop, mask, {})[0] is crop
    found = [view_crop(crop, mask, {"flip": flip, "rotate90": k})
             for flip in (False, True) for k in range(4)]
    assert len({a.tobytes() for a, _ in found}) == 8
    assert len({m.tobytes() for _, m in found}) == 8  # the mask turns with the crop
    assert np.array_equal(found[4][0], np.fliplr(crop))
    assert np.array_equal(found[1][0], np.rot90(crop))
    assert view_crop(crop, None, {"flip": True})[1] is None


@pytest.fixture
def cache(tmp_path):
    """Two plants on a colourful image, with a cut crop cache."""
    image = np.random.default_rng(0).integers(0, 255, (100, 120, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), image)
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 120, "height": 100,
                        "dataset": "acre"}],
            "categories": [{"id": 1, "name": "BEAVA"}, {"id": 2, "name": "CHEAL"}],
            "annotations": [
                {"id": 5, "image_id": 1, "category_id": 1, "bbox": [30, 20, 30, 30], "iscrowd": 0},
                {"id": 6, "image_id": 1, "category_id": 2, "bbox": [70, 40, 20, 20], "iscrowd": 0}]}
    folder = cache_dir(tmp_path, CROPS)
    return folder, read_index(write_cache(coco, tmp_path, folder / "val", CROPS)), image


def test_crop_dataset_views(cache):
    folder, rows, _ = cache

    def sample(view=None):
        return CropDataset(rows, folder / "val", ["BEAVA", "CHEAL"], CROPS, view=view)[0][0]

    plain = sample()
    assert torch.equal(sample({}), plain)
    assert torch.equal(sample({"flip": True}), plain.flip(2))
    assert torch.equal(sample({"rotate90": 2}), plain.flip(1).flip(2))
    tighter, looser = sample({"scale": 0.85}), sample({"scale": 1.15})
    assert tighter.shape == looser.shape == plain.shape
    assert not torch.equal(tighter, plain) and not torch.equal(looser, plain)
    with pytest.raises(ValueError, match="fixed view"):
        CropDataset(rows, folder / "val", ["BEAVA", "CHEAL"], CROPS,
                    {"flips": True}, {"flip": True})


def test_detection_input_views(cache):
    *_, image = cache
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection = {"bbox": [30, 20, 30, 30]}
    plain = detection_input(image, detection, CROPS)
    assert torch.equal(detection_input(image, detection, CROPS, {}), plain)
    assert torch.equal(detection_input(image, detection, CROPS, {"flip": True}), plain.flip(2))
    assert not torch.equal(detection_input(image, detection, CROPS, {"scale": 1.15}), plain)


def test_detection_view_matches_each_detection_whatever_the_order(cache, tmp_path):
    """Boxes of two images in mixed order: each row must get its own detection's probabilities."""
    cv2.imwrite(str(tmp_path / "b.png"), np.flipud(cache[2]).copy())
    images = [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}]
    boxes = [(2, [10, 10, 30, 30]), (1, [30, 20, 30, 30]), (2, [60, 40, 20, 20]),
             (1, [70, 40, 20, 20])]
    found = evaluation([("a", "BEAVA", "BEAVA")] * len(boxes))
    found.image_ids, found.boxes, found.gt = [i for i, _ in boxes], [b for _, b in boxes], {
        "images": images}
    torch.manual_seed(0)
    model = build_classifier(3, TINY_MODEL, 28)
    device = torch.device("cpu")
    probs = detection_view(model, ["a", "b", "c"], found, tmp_path, CROPS, {}, device, False, 2, 0)
    # classify_detections reports the same plants image by image
    detections = [{"image_id": i, "category_id": 1, "score": 0.9, "bbox": b} for i, b in boxes]
    results = classify_detections(detections, images, tmp_path, model, [1, 2, 3], CROPS, device,
                                  False, batch_size=2)
    for row, (image_id, box) in enumerate(boxes):
        result = next(r for r in results if r["image_id"] == image_id and r["bbox"] == box)
        assert result["class_scores"]["2"] == pytest.approx(probs[row, 1], abs=1e-4)
    assert len({tuple(p) for p in probs.round(5).tolist()}) == len(boxes)  # all different plants


def test_crop_view_gives_one_row_per_crop_in_order(cache):
    folder, rows, _ = cache
    torch.manual_seed(0)
    model = build_classifier(2, TINY_MODEL, 28)
    device = torch.device("cpu")
    both = crop_view(model, ["BEAVA", "CHEAL"], rows, folder / "val", CROPS, {}, device, False, 2, 0)
    alone = crop_view(model, ["BEAVA", "CHEAL"], rows[1:], folder / "val", CROPS, {}, device, False,
                      1, 0)
    assert both.shape == (2, 2)
    assert both[1] == pytest.approx(alone[0], abs=1e-5)


def test_mean_averages_and_a_vote_counts_top_classes():
    views = np.array([[[0.6, 0.4, 0.0]], [[0.3, 0.7, 0.0]], [[0.55, 0.45, 0.0]]])
    assert aggregate(views, "mean")[0] == pytest.approx([1.45 / 3, 1.55 / 3, 0])
    votes = aggregate(views, "vote")[0]
    assert votes == pytest.approx([2 / 3, 1 / 3, 0], abs=0.002)
    # the mean favours the second class, the vote the first: they can disagree
    assert votes.argmax() == 0 and aggregate(views, "mean")[0].argmax() == 1
    with pytest.raises(ValueError, match="method"):
        aggregate(views, "median")


def test_a_tied_vote_goes_to_the_larger_mean():
    views = np.array([[[0.51, 0.49]], [[0.1, 0.9]]])
    assert aggregate(views, "vote")[0].argmax() == 1


def test_a_gate_keeps_the_plain_view_of_plants_at_or_over_it():
    plain = np.array([[0.9, 0.1], [0.7, 0.3], [0.6, 0.4]])
    combined = np.array([[0.2, 0.8], [0.2, 0.8], [0.2, 0.8]])
    assert gated(plain, combined, None) is combined
    assert gated(plain, combined, 0.7).tolist() == [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]]


def test_as_saved_drops_small_probabilities_rounds_and_rescales():
    # as in test_twostage: 0.0001 is under CLASS_SCORE_MIN
    found = as_saved(np.array([[0.1, 0.6, 0.2999, 0.0001]]))[0]
    assert found[3] == 0
    assert found == pytest.approx(np.array([0.1, 0.6, 0.2999, 0]) / 0.9999)


def test_as_saved_is_what_the_pipeline_makes_of_a_detections_probabilities():
    """Raw probabilities through with_species and the scaling of action_scores: exactly the same,
    or the plain view wouldn't reproduce a normal evaluation."""
    raw = np.random.default_rng(0).dirichlet(np.full(69, 0.05), 200).astype(np.float32)
    ids = list(range(100, 169))
    for probs, saved in zip(raw, as_saved(raw)):
        scores_kept = with_species({"score": 0.9}, probs, ids)["class_scores"]
        total = sum(scores_kept.values())
        expected = np.array([scores_kept.get(str(i), 0.0) / total for i in ids])
        assert saved == pytest.approx(expected, abs=1e-6)


def test_shipped_config_is_usable():
    settings = load_config("configs/evaluate_tta.yaml")
    check_views(settings)
    check_scales(settings["views"], load_config("configs/classifier.yaml")["crops"])
    names = [variant.name for variant in variants(settings)]
    assert names[0] == PLAIN and "all 10" in names and "all 10, votes" in names
    assert len(names) == len(set(names))
    orientations = {(v.get("flip", False), v.get("rotate90", 0))
                    for v in settings["views"].values() if "scale" not in v}
    assert len(orientations) == 8


@pytest.mark.parametrize("change, message", [
    ({"views": {"flip": {"flip": True}}}, "plain"),
    ({"views": {PLAIN: {"rotate90": 1}}}, "plain"),
    ({"views": {**VIEWS, "odd": {"colour": 1}}}, "colour"),
    ({"views": {**VIEWS, "odd": {"rotate90": 4}}}, "quarter turns"),
    ({"views": {**VIEWS, "odd": {"scale": 0}}}, "positive"),
    ({"view_sets": {"three": [PLAIN, "nope"]}}, "nope"),
    ({"view_sets": {"three": ["flip", "looser"]}}, "must include"),
    ({"main_set": "nope"}, "main_set"),
])
def test_check_views_refuses_what_cant_work(change, message):
    with pytest.raises(ValueError, match=message):
        check_views({**SETTINGS, **change})


def test_a_window_larger_than_the_cache_is_refused():
    check_scales({"wide": {"scale": 1.5}}, CROPS)
    with pytest.raises(ValueError, match="cache_context"):
        check_scales({"wide": {"scale": 1.6}}, CROPS)


def test_variants_are_plain_each_set_and_the_main_set_by_vote_and_gate():
    assert variants(SETTINGS) == [
        Variant(PLAIN, [PLAIN]), Variant("two", [PLAIN, "flip"]),
        Variant("three", [PLAIN, "flip", "looser"]),
        Variant("three, votes", [PLAIN, "flip", "looser"], "vote"),
        Variant("three, under 0.7", [PLAIN, "flip", "looser"], "mean", 0.7)]
    assert len(variants({**SETTINGS, "votes": False, "gates": []})) == 3


def test_pick_is_fixed_and_in_order():
    assert pick(5, None).tolist() == [0, 1, 2, 3, 4]
    assert pick(5, 9).tolist() == [0, 1, 2, 3, 4]
    picked = pick(100, 10)
    assert picked.tolist() == pick(100, 10).tolist() == sorted(picked.tolist()) and len(picked) == 10


def test_views_are_kept_and_redone_only_if_they_are_of_other_plants_or_another_view(tmp_path):
    path, ids, probs = tmp_path / "x" / "flip.npz", np.array([5, 6]), np.array([[0.1, 0.9]] * 2)
    assert saved_view(path, ids, {"flip": True}) is None
    save_view(path, ids, {"flip": True}, probs, ["a", "b"])
    assert list(path.parent.iterdir()) == [path]  # no partial file left
    kept, classes = saved_view(path, ids, {"flip": True})
    assert kept.tolist() == probs.tolist() and classes == ["a", "b"]
    assert saved_view(path, np.array([5, 7]), {"flip": True}) is None
    assert saved_view(path, ids, {"flip": False}) is None


def test_collect_views_predicts_only_what_is_missing_and_orders_the_classes(tmp_path):
    ids = np.array([5, 6])
    calls = []

    def predict(view):
        calls.append(view)
        return np.array([[0.1, 0.9], [0.2, 0.8]]), ["b", "a"]

    views = {PLAIN: {}, "flip": {"flip": True}}
    found = collect_views(tmp_path, ids, views, predict, ["a", "b"])
    assert calls == [{}, {"flip": True}]
    assert found[PLAIN].tolist() == [[0.9, 0.1], [0.8, 0.2]]  # columns in the evaluation's order
    collect_views(tmp_path, ids, views, predict, ["a", "b"])
    assert len(calls) == 2  # all kept
    collect_views(tmp_path, ids, {**views, "new": {"rotate90": 1}}, predict, ["a", "b"])
    assert calls[2:] == [{"rotate90": 1}]  # a new view costs only that view
    with pytest.raises(SystemExit, match="other classes"):
        collect_views(tmp_path, ids, views, predict, ["a", "c"])


def test_crop_rows_keeps_a_copy_for_a_machine_without_the_cache(cache, tmp_path):
    _, rows, _ = cache
    run_config = {"data": {"dir": str(tmp_path)}, "crops": CROPS}
    kept = tmp_path / "out" / "crops" / "plants.csv"
    assert crop_rows(run_config, "val", kept) == rows
    other = {"data": {"dir": str(tmp_path / "nothing")}, "crops": CROPS}
    assert crop_rows(other, "val", kept) == rows
    kept.unlink()
    with pytest.raises(SystemExit, match="prepare_crops"):
        crop_rows(other, "val", kept)


PLANTS = [("a", "BEAVA", "BEAVA"), ("a", "CHEAL", "BEAVA"), ("a", "CHEAL", "CHEAL"),
          ("b", "ECHCG", "ECHCG"), ("b", "ECHCG", "CHEAL"), ("b", "BEAVA", None)]


def views_of(found):
    """Views of an evaluation: plain as it is; the flip right on plant 1, wrong on plant 2; the
    window as plain."""
    plain = found.probs.astype(np.float32)
    flip = plain.copy()
    flip[1], flip[2] = row_of("BEAVA", 0.95), row_of("ECHCG", 0.95)
    return {PLAIN: plain, "flip": flip, "looser": plain.copy()}


def test_the_plain_variant_gives_the_scores_of_the_evaluation_itself():
    found = evaluation(PLANTS)
    views = views_of(found)
    rows = evaluate_variants(found, views, SETTINGS)
    own = scores(found, as_saved(views[PLAIN]), SETTINGS)
    for group in (ALL, "a", "b"):
        row = next(r for r in rows if r["variant"] == PLAIN and r["dataset"] == group)
        assert row["species_right"] == own[group]["species_right"]
        assert row["harmful_at_0.5"] == own[group]["harmful_at_0.5"]
        assert row["wrong_to_right"] == row["right_to_wrong"] == 0
        assert row["partial_fit"] is None


def test_evaluate_variants_counts_what_changed_against_the_plain_view():
    found = evaluation(PLANTS)
    rows = evaluate_variants(found, views_of(found), SETTINGS)
    # two views: the mean follows the sure flip on plants 1 and 2
    row = next(r for r in rows if r["variant"] == "two" and r["dataset"] == ALL)
    assert (row["wrong_to_right"], row["right_to_wrong"]) == (1, 1)
    assert row["plants"] == 5  # the unlabelled detection isn't scored
    in_a = next(r for r in rows if r["variant"] == "two" and r["dataset"] == "a")
    in_b = next(r for r in rows if r["variant"] == "two" and r["dataset"] == "b")
    assert (in_a["wrong_to_right"], in_a["right_to_wrong"]) == (1, 1)
    assert (in_b["wrong_to_right"], in_b["right_to_wrong"]) == (0, 0)
    # a gate under the plain top probability (0.7): nothing changes
    gate = next(r for r in rows if r["variant"] == "three, under 0.7" and r["dataset"] == ALL)
    assert (gate["wrong_to_right"], gate["right_to_wrong"]) == (0, 0)


def test_a_vote_of_three_outvotes_the_one_sure_view():
    found = evaluation(PLANTS)
    views = views_of(found)
    # plain and looser agree on plant 1 (CHEAL), so the vote stays there; the mean moves less far
    vote = combine(Variant("v", [PLAIN, "flip", "looser"], "vote"), views)
    assert vote[1].argmax() == CLASSES.index("CHEAL")
    assert vote[1].max() == pytest.approx(2 / 3, abs=0.01)


def test_crop_evaluation_scores_labelled_boxes_and_leaves_partial_labels_to_their_fit():
    rows = [{"dataset": "a", "image_id": 1, "category": "BEAVA", "allowed": []},
            {"dataset": "a", "image_id": 1, "category": "CHEAL", "allowed": []},
            {"dataset": "b", "image_id": 2, "category": "ignore", "allowed": ["CHEAL", "ECHCG"]},
            {"dataset": "b", "image_id": 2, "category": "ignore", "allowed": ["ECHCG"]}]
    found = crop_evaluation(evaluation([]), rows)
    assert found.true.tolist() == [0, 1, -1, -1]
    assert found.plant_scores.tolist() == [1, 1, 1, 1]
    plain = np.array([row_of(name, 0.95) for name in ("BEAVA", "ECHCG", "CHEAL", "ECHCG")],
                     np.float32)
    settings = {**SETTINGS, "views": {PLAIN: {}}, "view_sets": {"one": [PLAIN]}, "main_set": "one",
                "votes": False, "gates": []}
    table = evaluate_variants(found, {PLAIN: plain}, settings, rows)
    row = next(r for r in table if r["variant"] == PLAIN and r["dataset"] == ALL)
    assert row["class_right"] == 0.5 and row["plants"] == 2
    assert row["crops_removed"] == 0  # a sure crop isn't removed
    assert row["partial_fit"] == 1.0
    assert partial_fit(rows, np.array([row_of("BEAVA")] * 4), CLASSES) == {"b": 0.0, ALL: 0.0}


def test_subset_evaluation_keeps_rows_together():
    found = subset_evaluation(evaluation(PLANTS), np.array([1, 4]))
    assert found.true.tolist() == [CLASSES.index("BEAVA"), CLASSES.index("CHEAL")]
    assert found.image_ids == [1, 4] and len(found.boxes) == 2 and found.probs.shape == (2, 4)
    assert [s[0] for s in found.sessions] == ["a", "b"]


def test_plain_checks_compare_with_the_normal_evaluations(tmp_path):
    found = evaluation(PLANTS[:2])
    (tmp_path / "eval" / "val_last").mkdir(parents=True)
    (tmp_path / "eval" / "val_last" / "metrics.json").write_text('{"accuracy": 0.9441}')
    table = [{"variant": PLAIN, "dataset": ALL, "class_right": 0.9441}]
    lines = plain_checks(tmp_path, found, found.probs, table, "val", "last", None)
    assert "on 2 of 2" in lines[0] and "difference 0.0" in lines[0]
    assert "class right 94.41%" in lines[1] and "accuracy 94.41%" in lines[1]
    # a subset can't be compared with the whole split's accuracy
    assert len(plain_checks(tmp_path, found, found.probs, table, "val", "last", 5)) == 1


def test_report_shows_ranges_across_runs_and_only_the_columns_that_apply(tmp_path):
    found = evaluation(PLANTS)
    rows = []
    for run in ("seed0", "seed1"):
        views = views_of(found)
        if run == "seed1":  # the flip is wrong on one more plant
            views["flip"][0] = row_of("ECHCG", 0.95)
        rows += [{"run": run, "plants_from": "detections", **row}
                 for row in evaluate_variants(found, views, SETTINGS)]
        rows += [{"run": run, "plants_from": "crops", **row, "partial_fit": 0.5}
                 for row in evaluate_variants(found, views, SETTINGS)]
    text = report(rows, SETTINGS, ["seed0", "seed1"], ["check line"])
    assert "Runs: seed0, seed1." in text and "- check line" in text
    assert "flip = flip True" in text and f"{PLAIN} = as it is" in text
    crops, detections = text.split("# Both stages")
    assert "partial-label fit" in crops and "partial-label fit" not in detections
    assert "## a: 3 plants" in detections and "## all: 5 plants" in detections
    assert detections.index("## all") < detections.index("## a:")
    two = next(line for line in detections.splitlines() if line.startswith("| two |"))
    assert two.endswith("| 1 | 1-2 |")  # wrong to right, then right to wrong as a range
    assert "| harmful at <=50% crops |" in crops


def test_csv_of_the_report_has_every_number(tmp_path):
    """write_csv gets rows of two kinds: all their columns must make it."""
    found = evaluation(PLANTS)
    rows = [{"run": "seed0", "plants_from": "crops", **row}
            for row in evaluate_variants(found, views_of(found), SETTINGS)]
    write_csv(tmp_path / "x.csv", rows)
    with open(tmp_path / "x.csv", newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == len(rows)
    assert {"run", "variant", "dataset", "species_right", "right_to_wrong"} <= set(written[0])
