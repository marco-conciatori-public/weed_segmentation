import numpy as np
import pytest

from weedseg.rows import (
    Geometry,
    RowSettings,
    find_pitch,
    plant_positions,
    session_geometry,
    split_rows,
)


def field(frames=20, rows=(0, 500), pitch=200, angle=0.0, jitter=0, across_jitter=0, seed=0):
    """Images 1 m wide of rows along `angle` (degrees), a plant every `pitch` mm."""
    rng = np.random.default_rng(seed)
    along, across = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]), np.array(
        [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
    images = []
    for frame in range(frames):
        start = 37.0 * frame
        points = [along * (start + k * pitch + rng.normal(0, jitter or 1e-9))
                  + across * (row + rng.normal(0, across_jitter or 1e-9))
                  for row in rows for k in range(int(1000 / pitch))]
        images.append(np.array(points))
    return images


def test_a_session_gives_direction_row_spacing_gap_and_pitch():
    geometry = session_geometry(field(angle=20, jitter=10))
    assert np.rad2deg(geometry.direction) == pytest.approx(20, abs=2)
    assert geometry.row_spacing == pytest.approx(500, abs=15)
    assert geometry.pitch == pytest.approx(200, abs=10)
    assert geometry.usual_gap == pytest.approx(200, abs=15)


def test_missing_plants_dont_change_the_pitch_nor_does_it_settle_on_a_fraction():
    rng = np.random.default_rng(1)
    images = [np.delete(image, rng.choice(len(image), 3, replace=False), axis=0)
              for image in field(jitter=5)]
    assert session_geometry(images).pitch == pytest.approx(200, abs=10)


def test_gaps_without_a_period_have_no_pitch_but_still_a_usual_gap():
    rng = np.random.default_rng(0)
    images = [np.column_stack([rng.uniform(0, 1000, 8), rng.normal(0, 5, 8)]) for _ in range(30)]
    geometry = session_geometry(images)
    assert geometry.pitch is None and geometry.regularity < RowSettings().min_regularity
    assert geometry.usual_gap is not None


def test_a_session_too_short_to_show_a_period_has_none():
    # four neat gaps could be chance
    geometry = session_geometry(field(frames=1, rows=(0,), pitch=200))
    assert geometry.gaps == 4 and geometry.pitch is None and geometry.usual_gap is None


def test_a_session_without_two_plants_in_an_image_has_no_geometry():
    assert session_geometry([np.array([[0.0, 0.0]]), np.empty((0, 2))]) is None


def test_two_plants_decide_the_direction_only_without_images_of_three():
    pair = np.array([[0.0, 0.0], [0.0, 300.0]])  # across the three-plant images' rows
    row = np.array([[0.0, 0.0], [300.0, 0.0], [600.0, 0.0]])
    assert np.rad2deg(session_geometry([pair]).direction) == pytest.approx(90)
    assert np.rad2deg(session_geometry([pair, row]).direction) == pytest.approx(0)


def test_split_rows_numbers_rows_in_order_and_keeps_chained_plants_together():
    rows = split_rows(np.array([1000.0, 0.0, 90.0, 180.0]), 100)
    assert rows.tolist() == [1, 0, 0, 0]


def test_the_wandering_centres_of_a_row_stay_one_row_once_the_spacing_is_known():
    images = field(rows=(0, 600), across_jitter=70, jitter=5)
    geometry = session_geometry(images)
    assert geometry.row_spacing == pytest.approx(600, abs=40) and geometry.same_row >= 250
    assert split_rows(images[0] @ np.array([0, 1.0]), geometry.same_row).max() == 1


def test_find_pitch_needs_distances_and_finds_multiples():
    assert find_pitch(np.array([]), RowSettings()) == (None, 0.0)
    pitch, fit = find_pitch(np.array([200.0, 400, 600, 200, 400, 200, 800]), RowSettings())
    assert pitch == pytest.approx(200, abs=5) and fit > 0.9


def geometry_of(pitch=None, usual_gap=None, same_row=100.0):
    scores = np.zeros(180)
    scores[0] = 1.0  # rows along x
    return Geometry(scores, 0.0, None, same_row, usual_gap, pitch, 0.5, 20)


ROW = np.array([[0.0, 0.0], [200.0, 0.0], [400.0, 0.0], [600.0, 0.0]])


def test_a_plant_is_judged_from_the_others_only():
    points = np.vstack([ROW, [[100.0, 300.0]]])
    found = plant_positions(points, np.array([True] * 5), geometry_of())
    # the plant on its own row of one: without it, that row isn't there
    assert found["row_distance"][4] == pytest.approx(300)
    assert found["on_row"][4] == 4 and found["strongest_row"][4] == 4
    # a plant of the row leaves three others on it, and the lone plant on another
    assert found["on_row"][0] == 3 and found["strongest_row"][0] == 3
    assert found["row_distance"][0] == 0


def test_a_plant_that_is_not_evidence_adds_nothing_to_a_row():
    points = np.vstack([ROW, [[100.0, 0.0]]])
    found = plant_positions(points, np.array([True] * 4 + [False]), geometry_of(usual_gap=200))
    assert found["on_row"][4] == 4  # the four evidence plants
    assert found["on_row"][0] == 3  # the fifth doesn't count for them


def test_spacing_says_how_close_the_next_plant_along_the_row_is():
    points = np.vstack([ROW, [[100.0, 0.0]]])
    found = plant_positions(points, np.array([True] * 4 + [False]), geometry_of(usual_gap=200))
    assert found["spacing"][4] == pytest.approx(0.5)  # half way between two crops
    assert found["spacing"][1] == pytest.approx(1)
    assert np.isnan(plant_positions(points, np.array([True] * 5), geometry_of())["spacing"]).all()


def test_phase_says_how_far_from_an_expected_plant():
    points = np.vstack([ROW, [[100.0, 0.0], [800.0, 0.0], [850.0, 0.0]]])
    found = plant_positions(points, np.array([True] * 4 + [False] * 3), geometry_of(pitch=200))
    assert found["phase"][4] == pytest.approx(0.5)  # half way between two crops
    assert found["phase"][5] == pytest.approx(0, abs=1e-9)  # where the next plant is expected
    assert found["phase"][6] == pytest.approx(0.25)
    assert found["phase"][0] == pytest.approx(0, abs=1e-9)  # from the other three
    assert np.isnan(plant_positions(points, np.ones(7, bool), geometry_of())["phase"]).all()


def test_an_image_with_plants_enough_can_correct_the_sessions_direction():
    tilted = ROW @ np.array([[np.cos(0.5), np.sin(0.5)], [-np.sin(0.5), np.cos(0.5)]])
    found = plant_positions(tilted, np.ones(4, bool), geometry_of())
    # within the 1 degree steps of the search; along the session's direction it would be 100s of mm
    assert found["row_distance"] == pytest.approx(0, abs=5)


def test_size_is_against_the_fullest_row_of_the_others():
    points = np.vstack([ROW, [[100.0, 500.0]]])
    sizes = np.array([100.0, 100, 100, 100, 50])
    found = plant_positions(points, np.ones(5, bool), geometry_of(), sizes)
    assert found["size_ratio"][4] == pytest.approx(0.5)
    assert found["size_ratio"][0] == pytest.approx(1)


def test_a_plant_with_no_others_stays_unknown():
    points = np.array([[0.0, 0.0], [50.0, 50.0]])
    found = plant_positions(points, np.array([True, False]), geometry_of())
    assert np.isnan(found["row_distance"][0])
    assert found["row_distance"][1] == pytest.approx(50)
    assert np.isnan(plant_positions(points, np.array([False, False]),
                                    geometry_of())["row_distance"]).all()
