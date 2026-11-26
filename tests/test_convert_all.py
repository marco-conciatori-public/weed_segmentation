from weedseg.data.convert_all import has_raw_data, setting_mismatches


def test_raw_data_needs_more_than_the_archives(tmp_path):
    raw = tmp_path / "raw"
    assert not has_raw_data(raw)
    raw.mkdir()
    assert not has_raw_data(raw)
    (raw / ".archives").mkdir()
    assert not has_raw_data(raw)
    (raw / "images").mkdir()
    assert has_raw_data(raw)


def test_setting_mismatches():
    sections = {"a": {"taxonomy": "species", "min_plant_size_mm": 20},
                "b": {"taxonomy": "species", "min_plant_size_mm": 20},
                "c": {"taxonomy": "species", "min_plant_size_mm": 30}}
    assert setting_mismatches(sections, ["taxonomy"]) == []
    assert setting_mismatches(sections, []) == []
    assert setting_mismatches(sections, ["taxonomy", "min_plant_size_mm"]) == [
        "min_plant_size_mm differs: 20 in a, b; 30 in c"]


def test_missing_setting_counts_as_none():
    assert setting_mismatches({"a": {"taxonomy": "species"}, "b": {}}, ["taxonomy"]) == [
        "taxonomy differs: 'species' in a; None in b"]
