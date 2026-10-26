import json

import yaml

from weedseg.compare import comparison, share, write_comparison


def make_run(root, name, seed, detail_p, accuracy, cross=None):
    """A classifier run folder with its val evaluation, and a two-stage evaluation made with
    it."""
    run = root / "classifier" / name
    folder = run / "eval" / "val_best"
    folder.mkdir(parents=True)
    config = {"run_name": name, "seed": seed, "augment": {"detail_p": detail_p, "flips": True}}
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [{"epoch": 0, "best": True}, {"epoch": 1, "best": True},
                                          {"epoch": 2}]), encoding="utf-8")
    metrics = {"accuracy": accuracy, "balanced_accuracy": 0.8}
    if cross is not None:
        metrics["cross_dataset_accuracy"] = cross
    (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (folder / "accuracy_by_dataset.csv").write_text(
        "dataset,crops,accuracy,balanced_accuracy\nacre,10,0.9,0.5\n", encoding="utf-8")
    (folder / "partial_labels.csv").write_text(
        "dataset,eppo,allowed,crops,label_fit,most_predicted\n"
        "oppd,1PLAK,x,30,0.5,\noppd,1COMF,y,10,1.0,\n", encoding="utf-8")
    both = root / "twostage" / f"det__{name}" / "eval" / "val"
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
    assert rows["best epoch"] == ["1", "1", "1"]
    assert rows["accuracy"] == ["90.0%", "91.0%", "92.0%"]
    assert rows["balanced, acre"] == ["50.0%"] * 3
    assert rows["crops removed, all"] == ["0.04%"] * 3
    assert rows["cross-dataset accuracy"] == ["-", "-", "60.0%"]
    assert rows["partial labels fit, oppd"] == ["62.5%"] * 3  # weighted by crops
    names = [row[0] for row in table]
    assert names.index("accuracy") < names.index("plants found, all") < names.index(
        "cross-dataset accuracy")


def test_write_comparison_and_missing_files(tmp_path):
    empty = tmp_path / "classifier" / "empty"
    empty.mkdir(parents=True)
    text = write_comparison([make_run(tmp_path, "base", 0, 0.0, 0.9), empty], "val", "best",
                            tmp_path / "twostage", tmp_path / "out" / "compare_1")
    assert "B: empty (no two-stage evaluation)" in text
    assert "| accuracy | 90.0% | - |" in text
    csv_text = (tmp_path / "out" / "compare_1.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == ",base,empty" and "**" not in csv_text
    assert (tmp_path / "out" / "compare_1.md").read_text(encoding="utf-8") == text
    assert share(None) == share("") == "-" and share(0) == "0.0%" and share("0.005") == "0.50%"
