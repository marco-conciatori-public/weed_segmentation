import json

import numpy as np
import yaml

from weedseg.compare import comparison, share, write_comparison
from weedseg.data.coco import encode_mask


def make_run(root, name, seed, detail_p, accuracy, cross=None, checkpoint="best",
             twostage_folder="val"):
    """A classifier run folder with its val evaluation of `checkpoint`, and a two-stage
    evaluation made with it (in eval/val as up to 2026-10-26, or `twostage_folder`)."""
    run = root / "classifier" / name
    folder = run / "eval" / f"val_{checkpoint}"
    folder.mkdir(parents=True)
    config = {"run_name": name, "seed": seed, "augment": {"detail_p": detail_p, "flips": True}}
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"epoch": 0, "val_accuracy": 0.8, "best": True},
        {"epoch": 1, "val_accuracy": 0.9, "best": True},
        {"epoch": 2, "val_accuracy": 0.89}]), encoding="utf-8")
    metrics = {"accuracy": accuracy, "balanced_accuracy": 0.8}
    if cross is not None:
        metrics["cross_dataset_accuracy"] = cross
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "accuracy_by_dataset.csv").write_text(
        "dataset,crops,accuracy,balanced_accuracy\nacre,10,0.9,0.5\n", encoding="utf-8")
    (folder / "partial_labels.csv").write_text(
        "dataset,eppo,allowed,crops,label_fit,most_predicted\n"
        "oppd,1PLAK,x,30,0.5,\noppd,1COMF,y,10,1.0,\n", encoding="utf-8")
    both = root / "twostage" / f"det__{name}" / "eval" / twostage_folder
    both.mkdir(parents=True)
    (both / "by_dataset.csv").write_text(
        "dataset,found,species_right,crops_removed\nall,0.8,0.9,0.0004\n", encoding="utf-8")
    return run


def test_comparison_puts_settings_overall_then_dataset_reliance(tmp_path):
    runs = [make_run(tmp_path, "base", 0, 0.0, 0.9), make_run(tmp_path, "seed1", 1, 0.0, 0.91),
            make_run(tmp_path, "detail", 0, 0.5, 0.92, cross=0.6)]
    notes, table = comparison(runs, "val", "best", tmp_path / "twostage")
    assert notes[0] == "A: base (both stages: det__base)"
    assert notes[-1].startswith("A and B differ only in seed")
    rows = {row[0]: row[1:] for row in table}
    assert table[0] == ["", "A", "B", "C"]
    assert rows["augment.detail_p"] == ["0.0", "0.0", "0.5"] and "augment.flips" not in rows
    assert "run_name" not in rows
    assert rows["epoch evaluated (best)"] == ["1", "1", "1"]
    assert rows["val accuracy, best epoch minus last"] == ["1.0%"] * 3
    assert rows["accuracy"] == ["90.0%", "91.0%", "92.0%"]
    assert rows["balanced, acre"] == ["50.0%"] * 3
    assert rows["crops removed, all"] == ["0.04%"] * 3
    assert rows["cross-dataset accuracy"] == ["-", "-", "60.0%"]
    assert rows["partial labels fit, oppd"] == ["62.5%"] * 3  # weighted by crops
    names = [row[0] for row in table]
    assert names.index("accuracy") < names.index("plants found, all") < names.index(
        "cross-dataset accuracy")


def box_mask(x, y):
    mask = np.zeros((100, 200), bool)
    mask[y:y + 30, x:x + 30] = True
    return encode_mask(mask)


def add_predictions(both, gt_path):
    """Ground truth and the two-stage predictions of an evaluation folder `both`: a maize
    plant removed up to defer_below 0.9 and one kept; goosefoots removed up to 0.95, 0.85
    (dataset a) and 0.6 (dataset b, no crops)."""
    plants = [(1, 1, 10, 1), (2, 1, 60, 1), (3, 1, 110, 2), (4, 1, 160, 2), (5, 2, 10, 2)]
    gt = {"images": [{"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 100,
                      "dataset": d} for i, d in [(1, "a"), (2, "b")]],
          "categories": [{"id": 1, "name": "ZEAMX"}, {"id": 2, "name": "CHEAL"},
                         {"id": 3, "name": "other_plant"}, {"id": 4, "name": "ignore"}],
          "annotations": [{"id": id_, "image_id": image, "category_id": category,
                           "bbox": [x, 10, 30, 30], "area": 900, "iscrowd": 0,
                           "segmentation": box_mask(x, 10)}
                          for id_, image, x, category in plants]}
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    scores = [{"2": 0.9, "1": 0.1}, {"1": 0.9, "2": 0.1}, {"2": 0.95, "1": 0.05},
              {"2": 0.85, "1": 0.15}, {"2": 0.6, "3": 0.4}]
    results = [{"image_id": image, "category_id": int(max(s, key=s.get)),
                "bbox": [x, 10, 30, 30], "score": 0.95, "plant_score": 0.95,
                "segmentation": box_mask(x, 10), "class_scores": s}
               for (_, image, x, _), s in zip(plants, scores)]
    (both / "predictions.json").write_text(json.dumps(results), encoding="utf-8")
    settings = {"score_threshold": 0.5, "iou_threshold": 0.5, "match_by": "mask",
                "rules": "configs/rules/example.yaml", "eppo_taxa": "configs/eppo_taxa.csv",
                "max_images": None}
    (both / "evaluation.json").write_text(
        json.dumps({"split_file": str(gt_path), "seed": 0, "settings": settings}),
        encoding="utf-8")


def test_decisions_at_the_same_share_of_crops_removed(tmp_path):
    run = make_run(tmp_path, "base", 0, 0.0, 0.9)
    add_predictions(tmp_path / "twostage" / "det__base" / "eval" / "val", tmp_path / "val.json")
    no_predictions = make_run(tmp_path, "seed1", 1, 0.0, 0.9)
    notes, table = comparison([run, no_predictions], "val", "best", tmp_path / "twostage",
                              crops_removed=[0.0, 0.5])
    rows = {row[0]: row[1:] for row in table}
    # no maize removed: up to 0.9 deferred, so only the goosefoot removed up to 0.95 is
    assert rows["harmful removed, up to 0.0% of crops, all"] == ["33.3% (0.901)", "-"]
    assert rows["harmful removed, up to 0.0% of crops, a"] == ["50.0% (0.901)", "-"]
    # one maize plant of two may go: the lowest defer_below removes every goosefoot
    assert rows["harmful removed, up to 50.0% of crops, a"] == ["100.0% (0.500)", "-"]
    assert rows["harmful removed, up to 0.0% of crops, b"] == ["100.0% (0.500)", "-"]
    names = [row[0] for row in table]
    assert names.index("crops removed, all") < names.index(
        "harmful removed, up to 0.0% of crops, all") < names.index("cross-dataset accuracy")
    (missing,) = [note for note in notes if note.startswith("B: no decisions")]
    assert "evaluation.json" in missing


def test_write_comparison_and_missing_files(tmp_path):
    empty = tmp_path / "classifier" / "empty"
    empty.mkdir(parents=True)
    text = write_comparison([make_run(tmp_path, "base", 0, 0.0, 0.9), empty], "val", "best",
                            tmp_path / "twostage", tmp_path / "out" / "compare_1")
    assert "B: empty (no two-stage evaluation)" in text
    assert "Settings that differ" not in text  # a run not trained yet differs in nothing
    assert "| accuracy | 90.0% | - |" in text
    csv_text = (tmp_path / "out" / "compare_1.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == ",base,empty" and "**" not in csv_text
    assert (tmp_path / "out" / "compare_1.md").read_text(encoding="utf-8") == text
    assert share(None) == share("") == "-" and share(0) == "0.0%" and share("0.005") == "0.50%"


def test_two_stage_evaluations_are_matched_by_checkpoint(tmp_path):
    run = make_run(tmp_path, "base", 0, 0.0, 0.9, checkpoint="last", twostage_folder="val_last")
    notes, table = comparison([run], "val", "last", tmp_path / "twostage")
    rows = {row[0]: row[1:] for row in table}
    assert notes[0] == "A: base (both stages: det__base)"
    assert rows["epoch evaluated (last)"] == ["2"]
    assert rows["plants found, all"] == ["80.0%"]
    # an evaluation of last.pt is not one of best.pt, and old eval/val folders are best.pt's
    assert comparison([run], "val", "best", tmp_path / "twostage")[0][0].endswith(
        "(no two-stage evaluation)")
    old = make_run(tmp_path, "old", 0, 0.0, 0.9)
    assert comparison([old], "val", "last", tmp_path / "twostage")[0][0].endswith(
        "(no two-stage evaluation)")
