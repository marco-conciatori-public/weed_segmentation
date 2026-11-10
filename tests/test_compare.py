import csv
import json

from weedseg.detector.compare import write_comparison
from weedseg.detector.run import eval_dir, resolve_run_dir, scale_dir


def config(tmp_path, name, sizes):
    return {"run_name": name, "output_dir": str(tmp_path / "runs"),
            "data": {"dir": "data/processed/public7/plant_min20mm"},
            "model": {"min_size": sizes},
            "evaluate": {"checkpoint": "best", "split": "val", "extra_splits": ["test_x"]},
            "evaluate_scale": {"split": "val"}}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_runs_side_by_side_with_missing_outputs_shown_as_dashes(tmp_path):
    base, scale = config(tmp_path, "base", [1088]), config(tmp_path, "scale", [640, 1088])
    run = resolve_run_dir(base)
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        json.dumps({"epoch": 1, "val_AP": 0.5, "val_AP50": 0.8, "best": True}) + "\n"
        + json.dumps({"epoch": 3, "val_AP": 0.4}) + "\n", encoding="utf-8")
    write_csv(eval_dir(base) / "by_dataset.csv", [
        {"dataset": "all", "images": 10, "plants": 100, "AP": 0.5, "AP50": 0.8, "found": 0.9,
         "missed": 10, "detections_on_no_plant": 5},
        {"dataset": "acre", "images": 4, "plants": 40, "AP": 0.4, "AP50": 0.7, "found": "",
         "missed": 4, "detections_on_no_plant": 2}])
    for c, found in ((base, 0.5), (scale, 0.75)):
        write_csv(scale_dir(c) / "report.csv", [
            {"dataset": "acre", "variant": "as_is", "target": "native", "found": 0.9},
            {"dataset": "acre", "variant": "as_is", "target": "1.0", "found": found},
            {"dataset": "acre", "variant": "rescaled", "target": "1.4", "found": 0.8}])

    text = write_comparison([base, scale], tmp_path / "compare")
    assert text == (tmp_path / "compare.md").read_text(encoding="utf-8")
    assert "| base | 1088 | 1 | 0.500 | 0.800 |" in text
    assert "| scale | 640-1088 | - | - | - |" in text
    assert "| all | 90.0 | - | 0.800 | - | 0.50 | - |" in text
    assert "| acre | - | - | 0.700 | - | 0.50 | - |" in text
    assert "## Evaluation: test_x" in text
    assert "| 1.0 | 50.0 / 75.0 |" in text
    assert "| rescaled from 1.4 | 80.0 / 80.0 |" in text
