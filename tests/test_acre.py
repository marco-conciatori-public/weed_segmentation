import numpy as np

from weedseg.data import acre

XML = """<root><metadata><size><width>40</width><height>20</height></size></metadata>
<data><clippings>
 <clipping id="1"><center><hasNoCenter>false</hasNoCenter><x>5</x><y>5</y></center>
  <points><point id="1" x="2" y="2"/><point id="2" x="9" y="2"/><point id="3" x="9" y="7"/>
   <point id="4" x="2" y="7"/></points>
  <class>weed</class><name>Chenopode </name><trusted><isTrusted>true</isTrusted></trusted>
 </clipping>
 <clipping id="2"><center><hasNoCenter>true</hasNoCenter></center>
  <points><point id="1" x="20" y="3"/><point id="2" x="25" y="6"/></points>
  <class>crop</class><name>mais</name><trusted><isTrusted>true</isTrusted></trusted>
 </clipping>
 <clipping id="3"><center><hasNoCenter>false</hasNoCenter></center>
  <points><point id="1" x="30" y="10"/><point id="2" x="35" y="10"/>
   <point id="3" x="35" y="15"/></points>
  <class>unknow</class><name>chenopode</name><trusted><isTrusted>false</isTrusted></trusted>
 </clipping>
 <clipping id="4"><points><point id="1" x="1" y="1"/><point id="2" x="2" y="1"/>
   <point id="3" x="2" y="2"/></points>
  <class>weed</class><name></name><trusted><isTrusted>true</isTrusted></trusted>
 </clipping>
 <clipping id="5"><points><point id="1" x="4" y="12"/><point id="2" x="8" y="12"/>
   <point id="3" x="8" y="15"/></points>
  <class>crop</class><name>Inconnu</name><trusted><isTrusted>true</isTrusted></trusted>
 </clipping>
</clippings></data></root>"""


def write_dataset(tmp_path):
    folder = tmp_path / "data" / "Maize" / "2022-06-08-16-44"
    folder.mkdir(parents=True)
    (folder / "rgb-1.xml").write_text(XML, encoding="utf-8")
    return tmp_path


def test_names_batches_and_labels(tmp_path):
    raw = write_dataset(tmp_path)
    assert acre.image_names(raw) == ["Maize/2022-06-08-16-44/rgb-1"]
    assert acre.batch_of("Maize/2022-06-08-16-44/rgb-1") == "Maize/2022-06-08-16-44"
    assert acre.box_labels(raw, "Maize/2022-06-08-16-44/rgb-1") == [
        ("chenopode", 8, 6), ("mais", 6, 4), ("unknow", 6, 6), ("unnamed (weed)", 2, 2),
        ("inconnu (crop)", 5, 4)]  # a plant without a name keeps the annotators' class


def test_records_masks_stems_and_ignore_regions(tmp_path):
    raw = write_dataset(tmp_path)
    (record,) = acre.iter_records(raw, ["Maize/2022-06-08-16-44/rgb-1"], 0.5)
    assert (record.file_name, record.width, record.height) == (
        "Maize/2022-06-08-16-44/rgb-1.jpg", 40, 20)
    assert record.group == "Maize/2022-06-08-16-44"
    assert record.attributes == {"crop_field": "maize", "batch": "2022-06-08-16-44",
                                 "mm_per_px": 0.5, "mm_per_px_estimated": True}
    weed, maize, unknown, unnamed, _ = record.instances
    assert weed.bbox_xyxy == (2, 2, 10, 8) and weed.stem_xy == (5.0, 5.0)
    assert weed.mask.sum() == 8 * 6 and not weed.ignore  # filled polygon, edges included
    assert maize.ignore and maize.stem_xy is None  # 2 points: no outline, box as mask
    assert np.array_equal(np.argwhere(maize.mask).min(axis=0), [3, 20])
    assert unknown.ignore and unknown.stem_xy is None  # untrusted; stem flag without x/y
    assert unnamed.source_label == "unnamed (weed)" and not unnamed.ignore
