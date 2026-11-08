# OPPD (Open Plant Phenotyping Database)

- Source: https://vision.eng.au.dk/open-plant-phenotyping-database/ (original host
  gitlab.au.dk/AUENG-Vision/OPPD, down in October 2026). We use Dataset Ninja's copy
  "OPPD: Full Images" in Supervisely format, one 45 GB tar (`scripts/download_datasets.py`,
  `configs/oppd.yaml`). Paper: Madsen et al., "Open Plant Phenotyping Database of Common
  Weeds in Denmark", Remote Sensing 2020.
- Owner: Aarhus University.
- Licence: **CC BY-NC-SA 4.0** (dataset page, and `LICENSE.md` in the download).
  Non-commercial and share-alike: like CropAndWeed, keep it out of any model meant for
  commercial use.

## Content

Measured from the download (2026-10-22): all 7,590 images and annotations are there, and
every JPEG is complete.

- **Trays, not fields.** Each image is one tray, photographed from above under the same
  rig, 4096×3000 px, rectified (black around the tray). The dataset page gives 44 px/mm²,
  i.e. **0.15 mm/px**, about 2.7 times finer than CropAndWeed: an image covers 62×45 cm.
- **One sown species per tray** (the image tag `eppo`). 168 trays in 4 trials: T01
  (May–June 2017, 60 trays), T02 (Sept–Oct 2017, 60), T03 (June–July 2018, 30), T04
  (June–Aug 2019, 18). Each species grew in 3 trays at a time; 8 species were grown twice
  (6 trays, mostly in two trials: CHEAL, CIRAR, CONAR, FUMOF, POLAV, SONOL, VERAR, VERPE)
  and EPHHE three times (9 trays). The dataset page says each species grew under 3 growth
  conditions; which tray had which is not in this copy.
- **Time series.** A tray is photographed on most days, ~48 images (up to 55), so its
  images show the same plants growing. The dataset page counts 64k plants in its 315k
  boxes, but Dataset Ninja's copy has no plant ids: the boxes of one plant can't be linked
  across days.
- **Boxes only**: 315,040, none with a mask. Box classes are EPPO codes:
  - the 46 sown species,
  - *Sonchus asper* (`SONAS`, 587 boxes, only in *Sonchus oleraceus* trays) and
    Asteraceae (`1COMF`, family only, 2,781 boxes, mostly in *S. oleraceus* trays),
  - `PPPDD` (37,991 boxes) and `PPPMM` (9,365): EPPO's "dicotyledonous" and
    "monocotyledonous plants", plants not identified to species, found in most trays,
  - `1PLAK` (1 box) and `3UNCLK` ("Unclassified", 2 boxes).
- **Mostly seedlings**: the median box is 7–43 mm depending on the species. Only 101,005
  boxes (32%) are at least 20 mm, in 5,733 of the 7,590 images.
- Download quirks: the `SINAR` image tags are garbled (Latin name "arvensis\tCharlock",
  English "Sinapis"), and two boxes are `[[0, 0], [0, 0]]`, no plant (dropped).

## Conversion choices

- `scripts/convert_oppd.py` (`configs/oppd.yaml`), shared pipeline `weedseg.data.convert`,
  reader `weedseg.data.oppd`.
- **Images without plants of at least 20 mm are dropped** (`keep_negative_images: false`):
  they are mostly the first days of each tray's time series, the same trays over and over.
- **Split by tray** (`group` = trial + box id, since box ids repeat across trials), balanced
  over species, **fractions 80/10/10**, which give 69/16/15% of the plants, close to the
  70/15/15 of the other datasets. The fractions are targets: with only 3 trays per species
  the balancing overshoots val and test, and 70/15/15 gave 58/21/21%, with 17 species
  split one tray each (decided 2026-10-22).
- `PPPDD`, `PPPMM` and `1PLAK` become `1PLAK`, plants of unknown species: plants for the
  detector like any other. `3UNCLK` gets no code: an ignore region. `1COMF` stays a family
  label.
- Brassica napus (`BRSNN`) has role `weed`: OPPD grows it as a weed (volunteer oilseed
  rape).
- mm/px is the dataset page's 0.1508 (`mm_per_px_estimated: false`).
- Image fields: `trial`, `tray`, `date`, `sown_species` (the tray's species).
- Supervisely rectangles include both corners, so the box is one pixel wider and taller
  than the difference of the corners.
- Label map: `configs/label_maps/oppd.csv`. `name_source = dataset`: the names in the
  image tags; `eppo`: from EPPO, for the codes found only on boxes and for the garbled
  `SINAR`.

| | Train | Val | Test |
|---|---|---|---|
| Images (with plants ≥ 20 mm) | 3,810 | 964 | 959 |
| Trays | 109 | 27 | 28 |
| Plants ≥ 20 mm | 69,562 | 16,439 | 15,004 |
| of which species unknown (`1PLAK`) | 3,862 | 1,221 | 616 |
| Plants under 20 mm and other ignore regions | 72,026 | 20,700 | 27,944 |

The 1,857 dropped images held another 93,363 boxes, all under 20 mm.

Every species has plants in train. Most species have 2 trays in train and 1 in val or
test; 7 are split one tray each. *Cyanus segetum* (`CENCY`) and *Geranium molle* (`GERMO`)
have all 3 trays in train, so OPPD scores neither on val or test. Some species got their
smallest tray for training: *Myosotis arvensis* has 72 train plants against 825 in val,
*Urtica urens* 187 against 1,281. The trays of a species differ a lot in how many plants
pass 20 mm, perhaps because of their growth conditions; the split balances images, not
plants.

## In the detector's data

OPPD is in `configs/combine.yaml` (`public5`). Its trays add many boxed plants, but little
variety: one rig, one site, even lighting, and the same plants day after day. Watch its
`by_dataset.csv` row and the other datasets' rows with and without it.

## Known limitations

- **Not field images**: trays of sieved soil under even lighting, one sown species each.
  What the detector learns of the background here is tray soil, not field.
- **Little variety per species**: 3 trays of one trial, one site, and the same plants on
  many days. No plant ids to cap images per plant; capping images per tray (e.g. one day in
  N) is possible if OPPD turns out to dominate.
- Not checked: whether plants that came up uninvited in a tray are boxed too.
