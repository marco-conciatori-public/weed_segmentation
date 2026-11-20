import pytest

from weedseg.data import filenames
from weedseg.data.coco import ImageRecord
from weedseg.data.convert import with_names_on_disk
from weedseg.data.filenames import on_disk, real_path


@pytest.fixture(autouse=True)
def fresh_listings():
    filenames._entries.cache_clear()
    yield
    filenames._entries.cache_clear()


def make_files(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_on_disk_gives_the_spelling_on_disk(tmp_path):
    make_files(tmp_path, "Images/IMG_1.JPG", "Images/img_2.jpg")
    assert on_disk("images/img_1.jpg", tmp_path) == "Images/IMG_1.JPG"
    assert on_disk("Images/img_2.jpg", tmp_path) == "Images/img_2.jpg"
    assert on_disk("Images/img_3.jpg", tmp_path) is None
    assert on_disk("Other/img_2.jpg", tmp_path) is None


def test_on_disk_prefers_the_exact_name_and_refuses_ambiguous_ones(tmp_path, monkeypatch):
    # Two names differing only in case can't coexist on Windows: fake the listing.
    monkeypatch.setattr(filenames.os, "listdir", lambda folder: ["a.jpg", "A.JPG"])
    assert on_disk("a.jpg", tmp_path) == "a.jpg"
    with pytest.raises(ValueError, match="A.JPG, a.jpg"):
        on_disk("a.Jpg", tmp_path)


def test_real_path_leaves_missing_files_as_they_are(tmp_path):
    make_files(tmp_path, "ann/x.xml")
    assert real_path(tmp_path / "ann" / "x.xml") == tmp_path / "ann" / "x.xml"
    assert real_path(tmp_path / "ann" / "y.xml") == tmp_path / "ann" / "y.xml"
    assert real_path(tmp_path / "none" / "y.xml") == tmp_path / "none" / "y.xml"


def test_converted_records_get_the_names_on_disk(tmp_path, capsys):
    make_files(tmp_path, "part/Images/IMG_1.JPG", "part/Images/IMG_2.jpg")
    records = [ImageRecord(name, 10, 10, "g") for name in
               ("part/images/IMG_1.jpg", "part/Images/IMG_2.jpg", "part/Images/IMG_3.jpg")]
    names = [r.file_name for r in with_names_on_disk(records, tmp_path)]
    # A missing image stays (the pipeline used to keep it too) and is reported.
    assert names == ["part/Images/IMG_1.JPG", "part/Images/IMG_2.jpg", "part/Images/IMG_3.jpg"]
    out = capsys.readouterr().out
    assert "1 image file names spelled as on disk" in out
    assert "1 images not found" in out and "IMG_3.jpg" in out
