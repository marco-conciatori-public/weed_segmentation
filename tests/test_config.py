from weedseg.config import load_config, merge


def test_merge_overrides_key_by_key():
    base = {"data": {"dir": "a", "num_workers": 6}, "steps": [1, 2], "seed": 0}
    merged = merge(base, {"data": {"dir": "b"}, "steps": [3]})
    assert merged == {"data": {"dir": "b", "num_workers": 6}, "steps": [3], "seed": 0}
    assert base["data"]["dir"] == "a"  # input untouched


def test_config_with_base_loads_on_top_of_it(tmp_path):
    (tmp_path / "base.yaml").write_text("run_name: x\ndata:\n  dir: species\n  workers: 2\n",
                                        encoding="utf-8")
    (tmp_path / "child.yaml").write_text(f"base: {tmp_path / 'base.yaml'}\ndata:\n  dir: plant\n",
                                         encoding="utf-8")
    assert load_config(tmp_path / "child.yaml") == {"run_name": "x",
                                                    "data": {"dir": "plant", "workers": 2}}


def test_detector_trains_on_what_combine_writes():
    combine, detector = load_config("configs/combine.yaml"), load_config("configs/detector.yaml")
    assert detector["data"]["dir"] == f"{combine['output_dir']}/{combine['name']}/plant_min20mm"
    assert load_config("configs/export_detector.yaml")["detector"] == "configs/detector.yaml"
    run = load_config("configs/run_detector.yaml")
    assert (run["combine"], run["detector"]) == ("configs/combine.yaml", "configs/detector.yaml")


def test_scale_variant_differs_only_in_its_run_name_and_training_sizes():
    run = load_config("configs/run_detector.yaml")
    assert run["detector_variants"] == ["configs/detector_scale.yaml"]
    assert "evaluate_scale" in run["steps"]
    detector, variant = (load_config("configs/detector.yaml"),
                         load_config("configs/detector_scale.yaml"))
    assert variant["run_name"] != detector["run_name"]
    sizes = variant["model"]["min_size"]
    # evaluated (and exported) at the size of the detector without scale augmentation
    assert sizes[-1] == max(sizes) == detector["model"]["min_size"][-1]
    same = {**variant, "run_name": detector["run_name"],
            "model": {**variant["model"], "min_size": detector["model"]["min_size"]}}
    assert same == detector
