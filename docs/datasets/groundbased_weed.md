# GROUNDBASED_WEED

- Source: DIGITAL.CSIC, https://doi.org/10.20350/digitalCSIC/17475. That server sent only
  the first 512 MiB of `TOMATO_1.zip` and `TOMATO_2.zip`, so the images and their boxes come
  from the authors' Nextcloud share (sent 2026-10-26), the same folders unpacked
  (`scripts/download_datasets.py`, `configs/groundbased_weed.yaml`; see
  [data-format.md](../data-format.md#downloading)). Paper: Moreno, Rivera Cárdenas and
  Andújar, "Ground-based imagery dataset for early weed classification in tomato crops",
  Data in Brief 2025 (https://doi.org/10.1016/j.dib.2025.112249, open on PMC as PMC12670878).
- Owner: Centre for Automation and Robotics, CSIC (Spain). Contact: d.andujar@csic.es.
- Licence: **ODbL 1.0** (`README.txt`). Attribution is required. An adapted version of the
  database that is used in public must stay under ODbL. Unlike CropAndWeed, PhenoBench and
  OPPD, it has no non-commercial clause.

## Content

Measured from the download (2026-10-26): all 1,216 images and their Pascal VOC files are
there (the abstract says 1,217), and every JPEG is complete.

- **Commercial tomato fields** in Santa Amalia (Badajoz, Spain), early in the season,
  photographed by hand from above with a Canon PowerShot SX540 HS. The paper says the
  photos were taken in May–June, 1–1.5 m above the canopy, in five fields covering 36 ha.
  It doesn't record which photo comes from which field.
- **TOMATO_1 = 2021**: 938 images, none with EXIF data.
  - 816 photos `IMG_<n>`: 453 are 5184×3888 and 363 are turned to 3888×5184 (the pixels
    themselves, not an EXIF flag).
  - 122 cut-outs of quadrat photos, `<quadrat>_IMG_<n>` (one is `25IMG_4626`). Each is
    3,100–4,100 px on a side and shows the inside of a white pipe frame with a numbered
    card.
    There are 99 quadrats, numbered 1–100. 23 of them have two cut-outs, and the two show
    different plants. The full photos aren't in the dataset.
- **TOMATO_2 = 2022**: 278 photos, 5184×3888. The EXIF data says the focal length was
  4.3 mm (the widest zoom), and that all of them were taken on 31 May 2022 in three
  sessions: 11:06–11:18 (52 photos), 12:08–12:25 (28) and 13:10–13:43 (198).
  - The VOC `folder` field names plots B and C ("media/alta densidad", 28 and 27 photos),
    but their photos are interleaved in time with the others, so the field isn't used.
- The camera numbers run on from TOMATO_1 (4004–5085) to TOMATO_2 (5128–5826), so it is
  one camera. Consecutive photos show different ground. Feature matching found an overlap
  in only 16 of the 1,215 pairs of camera-consecutive photos.
- **Boxes only**, one per plant: 20,991 in all. The class is an EPPO code, or `NR` (not
  recognised: the image wasn't clear enough or the plant was too small).

| Class | | TOMATO_1 | TOMATO_2 |
|---|---|---|---|
| `LYPES` | tomato (*Solanum lycopersicum*), crop | 1,075 | 0 |
| `SOLNI` | black nightshade (*Solanum nigrum*) | 2,657 | 3,665 |
| `CYPRO` | purple nutsedge (*Cyperus rotundus*) | 2,486 | 1,301 |
| `SETVE` | bristly foxtail (*Setaria verticillata*) | 1,255 | 535 |
| `POROL` | purslane (*Portulaca oleracea*) | 657 | 1,104 |
| `ECHCG` | barnyard grass (*Echinochloa crus-galli*) | 238 | 5,084 |
| `NR` | not recognised | 692 | 242 |

- **Not every plant has a box.** TOMATO_2's photos show the tomato rows, but no tomato is
  labelled. The paper says that plants cut by the image border, plants not clearly
  visible, and plants of uncertain species were left unlabelled. 515 boxes are flagged
  `truncated` and 45 `difficult`.
- **Box conventions vary.** The paper says the 2021 `CYPRO` boxes outline the whole plant
  and the 2022 ones only its centre. In the 2021 photos we looked at, the `CYPRO` boxes
  also cover only the centre, while the leaves reach well outside the box. A large box (e.g.
  a purslane) can hold smaller boxes of other plants. Some tomato boxes span a whole strip
  of the row (e.g. `IMG_4097`).
- `Annotations_extra/` repeats the boxes as COCO and YOLO files. They are not used.

### Ground resolution

The paper gives "approximately 0.05 cm/px" (0.5 mm/px). We use **0.3 mm/px**, an estimate
(`mm_per_px_estimated: true`), for three reasons:

- **The camera's geometry.** At 4.3 mm, with 1.19 µm pixels, 1–1.5 m gives 0.29–0.43 mm/px.
  0.5 mm/px would need 1.7 m.
- **The quadrat cut-outs measure 0.29 mm/px.** The frame spans about 3,500 px and its pipe
  about 110 px. Of the standard sizes, only a 1 m frame of 32 mm pipe has that ratio.
- **Plant sizes match.** `NR` plants have the same median box (~85 px) in the quadrats as in
  the other photos.

The value matters only for the 20 mm size filter. At 0.3 mm/px, 1,406 boxes fall under it;
at 0.5 mm/px, 190 would.

## Conversion choices

- `scripts/convert_groundbased_weed.py` (`configs/groundbased_weed.yaml`), shared pipeline
  `weedseg.data.convert`, reader `weedseg.data.groundbased_weed`. The VOC files are read;
  their sizes match the images.
- **Box only** (`convert_dataset(box_only=True)`): the images reach the species classifier,
  not Mask R-CNN ([data-format.md](../data-format.md#datasets-labelled-with-boxes-only)).
- **Groups are blocks of 10 consecutive camera numbers** (`group_block`), i.e. stretches of
  a walk through a field. A quadrat cut-out counts by its photo's number. The data records
  no field or date that could be used instead.
  - With blocks, val and test come from the same fields and days as train, so they measure
    how well the model handles new ground, not new fields.
  - Blocks of 10 cut 2 of the 16 overlapping pairs.
  - Larger blocks unbalance the split: with 50, val got 2% of the ECHCG plants.
- **Split 70/15/15, balanced over species, seed 5.** The balancing counts images, and most
  ECHCG plants are in dense TOMATO_2 photos (~40 boxes each), so the plant fractions depend
  on the seed. Seed 0 put 34% of the ECHCG plants in val. Seed 5, the best of seeds 0–9,
  keeps every species within 5 points of the fractions.
- `NR` becomes `1PLAK`, a plant of unknown species, with role `weed`. Once combined, these
  plants are ignore regions ([data-format.md](../data-format.md)).
- `LYPES` (tomato) has role `crop`. `configs/rules/example.yaml` lists it as a crop.
- Image fields: `subset` (`TOMATO_1`/`TOMATO_2`) and `year`, plus `quadrat` on the cut-outs.
  Annotation fields: `truncated` and `difficult` when the VOC file sets them. Those plants
  are kept as normal plants.
- Label map: `configs/label_maps/groundbased_weed.csv`. `name_source = readme`: the
  scientific names in `README.txt`.
- Every image keeps at least one plant of 20 mm or more, so none is dropped.

| | Train | Val | Test |
|---|---|---|---|
| Images (TOMATO_1 + TOMATO_2) | 835 (629 + 206) | 188 (158 + 30) | 193 (151 + 42) |
| Groups (blocks) | 93 | 21 | 22 |
| `SOLNI` | 4,072 | 675 | 786 |
| `ECHCG` | 3,866 | 581 | 735 |
| `CYPRO` | 2,492 | 597 | 640 |
| `SETVE` | 1,240 | 301 | 226 |
| `POROL` | 1,070 | 245 | 321 |
| `LYPES` | 738 | 178 | 159 |
| `1PLAK` (`NR`) | 472 | 82 | 109 |
| Plants ≥ 20 mm | 13,950 (71%) | 2,659 (14%) | 2,976 (15%) |
| Plants under 20 mm (ignore regions) | 1,059 | 121 | 226 |

The plants under 20 mm are mostly `SOLNI` (789) and `NR` (271).

## In the combination

Since 2026-10-26 GROUNDBASED_WEED is in `configs/combine.yaml` with CropAndWeed, ACRE,
PhenoBench and OPPD:
`data/processed/cropandweed+acre+phenobench+oppd+groundbased_weed/species_min50img_min20mm`,
the data of the two-stage configs. Box-only images (OPPD and GROUNDBASED_WEED) are 4,645 of
12,233 in train, 1,152 of 3,359 in val and 1,152 of 2,570 in test. Compared with the four
datasets without it (`min_train_images: 50`):

- **Species classes grow from 64 to 68.** None is lost. The new ones are `LYPES`, `CYPRO`,
  `SETVE` and `POROL`. The first three have training plants from GROUNDBASED_WEED only.
- `POROL` takes CropAndWeed's 27 and ACRE's 9 training plants out of `other_plant`. As with
  OPPD, CropAndWeed's `by_dataset` rows then stop being comparable with earlier runs.
- Existing classes gain training plants: `ECHCG` goes from 813 to 4,679 and `SOLNI` from
  2,555 to 6,627.
- Training images grow from 11,398 to 12,233, all 835 new ones box-only.
- **The detector's data doesn't change.** Box-only images are left out of it, and the plant
  copy's images with masks and their annotations are identical with and without
  GROUNDBASED_WEED (checked 2026-10-26). Only the classifier needs retraining; the runner
  still retrains the detector, since its data folder has a new name.

## Known limitations

- **One site, one camera, two seasons.** The classifier learns tomato, *Cyperus rotundus*
  and *Setaria verticillata* from this dataset alone, so it may learn the dataset's look
  (soil, light, camera) along with the plants. `SETVE` against `SETVI` (green foxtail,
  CropAndWeed only) is the pair to watch. Scores on other datasets would show it.
- The val and test splits share fields and days with train (see the groups above).
- **Unlabelled plants**, tomatoes in all of TOMATO_2 among them. The classifier is
  unaffected, since it sees only the boxes. Scoring detections on this dataset would need
  them labelled or masked out.
- Box conventions differ (see `CYPRO` above). The crops of a partly boxed plant show only
  part of it.
- The ground resolution is an estimate (see above).
