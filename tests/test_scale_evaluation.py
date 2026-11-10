import json

import cv2
import numpy as np
import pytest

from weedseg.scale_evaluation import (
    IGNORE,
    NATIVE,
    PLANT,
    View,
    band_names,
    cases,
    choose_images,
    dataset_names,
    detect_all,
    in_window,
    plan,
    report,
    saved_check,
    score,
    score_all,
    show,
    to_image,
)

SIZES = (1000, 2000)  # the detector's min_size and max_size
SETTINGS = {"targets": [0.8, 4.0], "rescaled": [4.0], "deployed": True, "max_upsample": 1.5,
            "score_threshold": 0.5, "iou_threshold": 0.5, "size_bands_mm": [40, 80]}


def test_plan_puts_each_view_at_its_target():
    image = {"width": 2000, "height": 1000, "mm_per_px": 0.5}  # usual resize 1: 0.5 mm/px
    assert plan(image, "as_is", NATIVE, SIZES, 1.5) == View(1.0)
    assert plan(image, "as_is", 1.0, SIZES, 1.5) == View(0.5)  # 0.5 / 0.5 = 1 mm/px
    close = plan(image, "as_is", 0.4, SIZES, 1.5)  # a centred window, enlarged 1.25 times
    assert close == View(1.25, (200, 100, 1800, 900))
    assert show(np.zeros((1000, 2000, 3), np.uint8), close).shape == (1000, 2000, 3)
    assert plan(image, "as_is", 0.25, SIZES, 1.5) is None  # would enlarge 2 times
    assert plan(image, "rescaled", 1.0, SIZES, 1.5) == View(1.0, degrade=0.5)
    assert plan(image, "rescaled", 0.4, SIZES, 1.5) is None  # only coarser than native
    assert plan(image, "rescaled", NATIVE, SIZES, 1.5) is None
    # at a resize of 1 the deployed view is the same view, so it is predicted once
    assert plan(image, "deployed", NATIVE, SIZES, 1.5).key == View(1.0).key
    assert plan(image, "deployed", 1.0, SIZES, 1.5) is None

    large = {"width": 4000, "height": 2000, "mm_per_px": 0.25}  # usual resize 0.5
    assert plan(large, "deployed", NATIVE, SIZES, 1.5) == View(0.5, deployed=True)
    assert plan(large, "as_is", 1.0, SIZES, 1.5) == View(0.25)
    # its own pixels are fine enough for a close-up: a window at 0.625 is still a shrink
    assert plan(large, "as_is", 0.4, SIZES, 1.5) == View(0.625, (400, 200, 3600, 1800))


def test_show_and_to_image_put_detections_back_in_place():
    image = np.zeros((1000, 2000, 3), np.uint8)
    view = View(1.25, (200, 100, 1800, 900))
    shown = show(image, view)
    found = to_image(np.array([[250, 125, 125, 62.5, 0.9]]), view, (2000, 1000), (1600, 800))
    assert np.allclose(found, [[400, 200, 100, 50, 0.9]])
    assert show(image, View(0.5, deployed=True)) is image  # the detector resizes it
    assert show(image, View(1.0, degrade=0.25)).shape == shown.shape


def test_in_window_keeps_whole_plants_and_ignores_cut_ones():
    boxes = [{"id": 1, "bbox": [20, 20, 10, 10], "category_id": PLANT, "iscrowd": 0},
             {"id": 2, "bbox": [95, 20, 10, 10], "category_id": PLANT, "iscrowd": 0},
             {"id": 3, "bbox": [200, 20, 10, 10], "category_id": PLANT, "iscrowd": 0}]
    kept = in_window(boxes, (10, 10, 100, 100))
    assert [a["id"] for a in kept] == [1, 2]
    assert kept[0] is boxes[0]
    assert kept[1]["bbox"] == [95, 20, 5, 20 - 10] and kept[1]["iscrowd"] == 1
    assert kept[1]["category_id"] == IGNORE


def test_choose_images_stops_at_enough_plants_and_is_fixed():
    images = [{"id": k, "file_name": f"{k}.png", "width": 10, "height": 10, "mm_per_px": 1.0,
               "dataset": "a" if k < 5 else "b"} for k in range(8)]
    annotations = [{"id": 10 * k + j, "image_id": k, "category_id": 1, "iscrowd": int(j == 2)}
                   for k in range(8) for j in range(3)]  # two plants and a crowd region each
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "plant"}]}
    assert dataset_names(coco) == ["a", "b"]
    chosen = choose_images(coco, ["a", "b"], 3, 1, 0)
    assert [i["dataset"] for i in chosen] == ["a", "b"]  # max one image each
    chosen = choose_images(coco, ["a", "b"], 3, None, 0)
    assert [i["dataset"] for i in chosen] == ["a", "a", "b", "b"]  # 4 plants >= 3
    assert chosen == choose_images(coco, ["a", "b"], 3, None, 0)
    with pytest.raises(SystemExit):
        choose_images(coco, ["c"], 3, None, 0)


def squares(image: np.ndarray, deployed: bool) -> np.ndarray:
    """A detector that finds white squares of at least 10 px."""
    count, _, stats, _ = cv2.connectedComponentsWithStats((image[..., 0] > 127).astype(np.uint8))
    return np.array([[x, y, w, h, 0.9] for x, y, w, h, _ in stats[1:count]
                     if w >= 10 and h >= 10], np.float32).reshape(-1, 5)


def fails(image: np.ndarray, deployed: bool) -> np.ndarray:
    raise AssertionError("predicted again")


def field(folder):
    """An image of 24 px white squares on black (1 mm/px), saved in `folder`; with its truth,
    the detector's sizes (a usual resize of 1) and the views of SETTINGS."""
    rgb = np.zeros((400, 800, 3), np.uint8)
    boxes = [(x, y) for x in range(40, 760, 80) for y in range(40, 360, 80)]
    truth = {1: []}
    for k, (x, y) in enumerate(boxes):
        rgb[y:y + 24, x:x + 24] = 255
        truth[1].append({"id": k, "image_id": 1, "bbox": [x, y, 24, 24], "area": 576,
                         "iscrowd": 0, "category_id": PLANT})
    cv2.imwrite(str(folder / "field.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    image = {"id": 1, "file_name": "field.png", "width": 800, "height": 400, "mm_per_px": 1.0,
             "dataset": "d"}
    sizes = (400, 800)
    plans = {(variant, target): {1: plan(image, variant, target, sizes, 1.5)}
             for variant, target in cases(SETTINGS)}
    return image, truth, sizes, plans


def test_a_detector_needing_pixels_loses_plants_when_coarse_and_rescaling_brings_them_back(
        tmp_path):
    image, truth, sizes, plans = field(tmp_path)
    boxes = truth[1]
    cache = tmp_path / "cache.jsonl"
    detections = detect_all([image], plans, tmp_path, squares, cache)
    rows = score_all([image], truth, plans, detections, SETTINGS, ["d"], sizes, 300)
    by_case = {(r["variant"], r["target"]): r for r in rows}
    assert by_case["as_is", "native"]["found"] == 1.0
    assert by_case["as_is", "native"]["plants"] == len(boxes)
    assert by_case["deployed", "native"]["found"] == 1.0
    assert by_case["as_is", 4.0]["found"] == 0.0  # 6 px squares
    assert by_case["rescaled", 4.0]["found"] == 1.0  # enlarged back to 24 px
    close = by_case["as_is", 0.8]  # a window enlarged 1.25 times: the cut squares are left out
    assert close["found"] == 1.0 and 0 < close["plants"] < len(boxes)
    assert close["false_per_image"] == 0.0

    again = detect_all([image], plans, tmp_path, fails, cache)  # all from the cache
    assert again.keys() == detections.keys()
    assert all(np.allclose(again[k], detections[k], atol=0.06) for k in again)

    text = report(rows, SETTINGS, ["a check"], ["d"], "run/best.pt", [300, 400], sizes)
    assert "| native | | 100.0 (1.00) |" in text and "| 4 | 1.60 m | 0.0 |" in text
    assert "300-400 px (scale augmentation)" in text and "- a check" in text


def test_saved_check_compares_the_native_rows_with_the_saved_evaluation(tmp_path):
    image, truth, _, plans = field(tmp_path)
    detections = detect_all([image], plans, tmp_path, squares, tmp_path / "cache.jsonl")
    assert "no saved evaluation" in saved_check(tmp_path / "eval", [image], truth, plans,
                                                detections, SETTINGS, ["d"], 300)[0]
    (tmp_path / "eval").mkdir()
    half = [{"image_id": 1, "category_id": 1, "bbox": a["bbox"], "score": 0.9}
            for a in truth[1][::2]]
    (tmp_path / "eval" / "predictions.json").write_text(json.dumps(half))
    line, = saved_check(tmp_path / "eval", [image], truth, plans, detections, SETTINGS, ["d"],
                        300)
    assert line.startswith("d (36 plants of 1 images): found 50.0% in the saved evaluation")
    assert "100.0% here (deployed resize)" in line


def test_a_line_cut_short_by_a_killed_run_is_predicted_again(tmp_path):
    image, _, _, plans = field(tmp_path)
    cache = tmp_path / "cache.jsonl"
    found = detect_all([image], plans, tmp_path, squares, cache)
    lines = cache.read_text().splitlines()
    cache.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])  # killed while writing
    again = detect_all([image], plans, tmp_path, squares, cache)
    assert again.keys() == found.keys()
    assert detect_all([image], plans, tmp_path, fails, cache).keys() == found.keys()


def test_score_counts_false_detections_but_not_those_on_ignore_regions():
    image = {"id": 1, "width": 100, "height": 100, "mm_per_px": 2.0}
    truth = {1: [{"id": 1, "image_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0,
                  "category_id": PLANT},
                 {"id": 2, "image_id": 1, "bbox": [60, 60, 30, 30], "area": 900, "iscrowd": 1,
                  "category_id": IGNORE}]}
    detections = {1: np.array([[10, 10, 30, 30, 0.9], [62, 62, 25, 25, 0.9],
                               [60, 10, 20, 20, 0.9], [10, 60, 20, 20, 0.3]], np.float32)}
    values = score([image], truth, {1: View(1.0)}, detections, SETTINGS, 300)
    assert values["plants"] == 1 and values["found"] == 1.0
    assert values["false_per_image"] == 1.0 and values["precision"] == 0.5
    assert values["found 40-80 mm"] == 1.0 and values["found under 40 mm"] is None  # 60 mm
    assert values["plant_px"] == 30


def test_band_names():
    assert band_names([40, 80]) == ["under 40 mm", "40-80 mm", "80 mm and over"]
