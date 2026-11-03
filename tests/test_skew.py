import numpy as np
import pytest

from weedseg.skew import skew_probabilities

CLASSES = ["BEAVA", "ZEAMX", "CHEAL", "ECHCG", "other_plant"]
CROPS = {"BEAVA", "ZEAMX"}
PROBS = np.array([[0.1, 0.1, 0.4, 0.3, 0.1]])


def skew(crop_list=(), weed_list=(), probs=PROBS, multiplier=5.0, classes=CLASSES, crops=CROPS):
    return skew_probabilities(probs, classes, crops, crop_list, weed_list, use_lists=True,
                              multiplier=multiplier)


def test_off_by_default_and_with_empty_lists():
    assert skew_probabilities(PROBS, CLASSES, CROPS, ["BEAVA"], ["ECHCG"]) == pytest.approx(PROBS)
    assert skew() == pytest.approx(PROBS)
    assert skew(multiplier=100) == pytest.approx(PROBS)


def test_the_input_is_left_as_it_was():
    probs = PROBS.copy()
    skew(["BEAVA"], ["ECHCG"], probs)
    assert probs == pytest.approx(PROBS)


def test_a_weed_list_moves_probability_between_weeds_only():
    # CHEAL 0.4 / 5 = 0.08, ECHCG stays 0.3, scaled back to the weeds' total of 0.7
    expected = [0.1, 0.1, 0.08 * 0.7 / 0.38, 0.3 * 0.7 / 0.38, 0.1]
    assert skew(weed_list=["ECHCG"])[0] == pytest.approx(expected)


def test_a_crop_list_moves_probability_to_all_other_classes():
    assert skew(crop_list=["BEAVA"])[0] == pytest.approx(
        np.array([0.1, 0.02, 0.4, 0.3, 0.1]) / 0.92)


def test_both_lists_are_applied_in_turn():
    weeds_done = [0.1, 0.02, 0.08 * 0.7 / 0.38, 0.3 * 0.7 / 0.38, 0.1]
    assert skew(["BEAVA"], ["ECHCG"])[0] == pytest.approx(np.array(weeds_done) / 0.92)


def test_a_missed_weed_can_stay_on_top_but_a_listed_look_alike_can_beat_it():
    probs = np.array([[0.05, 0.05, 0.6, 0.2, 0.1]])
    assert CLASSES[skew(weed_list=["ECHCG"], probs=probs).argmax()] == "ECHCG"  # 0.12 < 0.2
    assert CLASSES[skew(weed_list=["ECHCG"], probs=probs, multiplier=2).argmax()] == "CHEAL"


def test_a_weed_list_doesnt_push_a_plant_towards_the_crop():
    probs = np.array([[0.3, 0.0, 0.4, 0.2, 0.1]])
    out = skew(weed_list=["ECHCG"], probs=probs)[0]
    assert out[:2].sum() == pytest.approx(0.3) and out[4] == pytest.approx(0.1)
    assert CLASSES[out.argmax()] == "ECHCG"
    # dividing without scaling back would have made this weed a crop: 0.3 > 0.2 and 0.08


def test_other_plant_is_never_divided():
    out = skew(weed_list=["ECHCG"], crop_list=["BEAVA"])[0]
    assert out[4] == pytest.approx(0.1 / 0.92)  # only scaled, like every class
    with_other = skew(weed_list=["other_plant"])[0]
    assert with_other[4] == pytest.approx(0.1)
    assert with_other[2:4] == pytest.approx([0.4, 0.3])  # both divided, then scaled back


def test_the_lists_override_the_crop_tags():
    # volunteer maize on the weed list: not divided, and it counts among the weeds
    out = skew(crop_list=["BEAVA"], weed_list=["ZEAMX"])[0]
    weeds = np.array([0.1, 0.4 / 5, 0.3 / 5])
    weeds *= 0.8 / weeds.sum()
    assert out == pytest.approx(np.array([0.1, *weeds, 0.1]))
    # a crop the tags don't know: on the crop list it keeps its probability; the rest are weeds
    out = skew(crop_list=["CHEAL"], weed_list=["ECHCG"], crops=set())[0]
    weeds = np.array([0.1 / 5, 0.1 / 5, 0.3])
    weeds *= 0.5 / weeds.sum()
    assert out == pytest.approx(np.array([weeds[0], weeds[1], 0.4, weeds[2], 0.1]))


def test_names_the_model_has_no_class_for_have_no_effect():
    assert skew(crop_list=["BEAVA", "XXXXX"], weed_list=["ECHCG", "YYYYY"]) == pytest.approx(
        skew(crop_list=["BEAVA"], weed_list=["ECHCG"]))


def test_a_list_of_nothing_the_model_knows_still_divides_the_rest():
    weeds = np.array([0.4, 0.3]) / 5
    assert skew(weed_list=["YYYYY"])[0][2:4] == pytest.approx(weeds * 0.7 / weeds.sum())


def test_rows_sum_to_one_and_zero_rows_stay_zero():
    probs = np.array([[0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0], [0.05, 0.05, 0.3, 0.3, 0.1]])
    out = skew(["BEAVA"], ["ECHCG"], probs)
    assert out.sum(1) == pytest.approx([1, 1, 0, 1])
    assert np.isfinite(out).all()


def test_the_multiplier_sets_the_strength_and_must_be_at_least_one():
    assert skew(["BEAVA"], ["ECHCG"], multiplier=1) == pytest.approx(PROBS)
    weak, strong = skew(weed_list=["ECHCG"], multiplier=2), skew(weed_list=["ECHCG"], multiplier=10)
    assert weak[0, 3] < strong[0, 3]
    with pytest.raises(ValueError, match="at least 1"):
        skew(["BEAVA"], multiplier=0.5)
