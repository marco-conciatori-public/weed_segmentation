# VCD (Vegetable Crops Dataset)

- Source: Weed-AI's copy, three uploads, one per crop: beans
  (https://weed-ai.sydney.edu.au/datasets/499647c5-fdd2-472f-bccb-ee4acda631c9), maize
  (f6e9953d-dc9c-4bf2-8309-e17d726030d1) and leeks (2adbf2e1-ebc1-48fd-bbbc-76ef8b97e255).
  `scripts/download_datasets.py`, `configs/vcd.yaml`; format in [weedai.md](weedai.md).
  Original: Mendeley Data "Vegetable Crops Dataset for Proximal Sensing (VCD)", doi
  10.17632/d7kbzjr83k.1 (one 2.7 GB `DB.zip`, Pascal VOC, with the authors' train/valid
  lists).
- Paper: Lac, Keresztes, Louargant, Donias and Da Costa, "An annotated image dataset of
  vegetable crops at an early stage of growth for proximal sensing applications", Data in
  Brief 42 (2022) 108035 (open on PMC as PMC8933512).
- Owner: Université de Bordeaux (IMS, Bordeaux Sciences Agro, ENSEIRB-Matmeca) with CTIFL.
- Licence: **CC BY 4.0** (Mendeley Data record, and Weed-AI). Commercial use is allowed with
  attribution.

## Content

From the paper, and measured from the Weed-AI download (2026-10-30):

- **Maize, bean and leek** 2 to 5 weeks after sowing or planting, in rows that were "not
  hoed or treated". Photographed by an acquisition module under a hull, lit by two LED
  panels, camera facing the ground at 35-45 cm (Basler acA2500 with a 4 mm lens in the
  current module; earlier ones not named). Parts of the hull (black curtain, weeding tools)
  show at the image edges.
- **Four sites in France**, and the dates in the file names
  (`<date>_<site>_<crop>_<row>_<index>`):

  | Site (file name) | Institution | Dates | Images (beans / maize / leeks) |
  |---|---|---|---|
  | Montoldre (`montoldre`) | INRAE | 2019-05-23, 2019-09-25, 2020-10-12 | 220 / 439 / 0 |
  | Prigonrieux (`lanxade`) | CTIFL | 2019-10-05, 2020-10-01, 2021-07-20 | 387 / 352 / 113 |
  | Liposthey (`liposthey`) | Fermes Larrère | 2019-07-03, 2021-03-29 | 0 / 0 / 433 |
  | Bordeaux (`bordeaux`) | Bordeaux Sciences Agro | 2021-05-24 | 0 / 0 / 55 |
  | undated, unknown site (`misc_<crop>_<n>`) | | | 163 / 310 / 0 |

- **Same site as ACRE.** INRAE Montoldre is where ACRE was recorded (June 2022), and VCD's
  sown weeds are ACRE's: mustard, ryegrass, chamomile and fat-hen, besides the natural
  purslane, black nightshade and couch grass. VCD's Montoldre images are from 2019-2020, with
  another camera module.
- **Only the crops are boxed**, each twice: the whole plant and its stem entry point. The
  weeds are visible but have no box. 2,472 images (770 beans, 1,101 maize, 601 leeks), every
  one with at least one crop: 2,907 beans, 2,259 maize and 3,068 leeks (the paper's
  counts differ slightly: 779 / 1,065 / 601 images).
- **Image sizes vary**: 2048x1536, 1024x768 (half that), 1440x1080, 1280x960, and for the
  undated images 2448x2048 or 632x632 (whole frames, downscaled). The paper gives no
  resolution and no ground resolution.
- A row and day can sit in two uploads: `2020-10-01_lanxade_bean_2` has 72 images in the
  bean upload and 43 in the maize one (mixed plots); the images differ.

## Conversion choices

- `scripts/convert_weedai.py` (`configs/vcd.yaml`, listed in `configs/weedai.yaml`), reader
  `weedseg.data.weedai`, shared pipeline `weedseg.data.convert`. **Box only**: the classifier
  learns from these crops, Mask R-CNN doesn't see them.
- **Stem boxes are left out** (`drop_labels`). They could become stem points later
  (`Instance.stem_xy`); nothing uses stem points yet.
- **Group = one crop row on one day** (the file name up to the image index; the undated ones
  by `misc_<crop>_<n>`), across uploads: 62 groups. Neighbouring images of a row may show
  the same plants. Val and test share days and plots with train, not plants.
- **Scale per image** (`footprint_mm_estimate: 570`, the ground length of the longer side):
  2048 px → 0.28 mm/px, 1024 → 0.56, 2448 → 0.23, 632 → 0.90. Estimate: 2592 px of 2.2 µm
  behind a 4 mm lens at ~40 cm see ~570 mm. The undated images' camera is unknown; the same
  footprint is assumed. It matters only for the 20 mm size filter: 4 crops fall under it,
  and one image is dropped for it.
- Leek is `ALLPO` (*Allium porrum*), role crop. The dataset says *Allium ampeloprasum*, the
  species that also covers wild leek; the label map's `eppo_note` explains it.
  `configs/rules/example.yaml` lists leek as a crop.
- **Split 70/15/15 by group, balanced over species, seed 5.** Seed 0 put 26% of the leeks
  in val (one leek row has 206 images); seed 5, the best of 0-9, keeps every crop within 5
  points of the fractions. The authors' train/valid lists are not used: they have no test
  list, and Weed-AI's copy doesn't carry them.

| | Train | Val | Test |
|---|---|---|---|
| Images | 1,690 | 373 | 408 |
| Groups (rows) | 46 | 9 | 7 |
| `ALLPO` leek | 2,213 | 344 | 511 |
| `PHSVX` bean | 1,980 | 526 | 401 |
| `ZEAMX` maize | 1,461 | 437 | 361 |

## Known limitations

- **No weeds.** The classifier sees only crops from this rig, so its look (hull, LED light)
  may come to mean "crop". The weeds in these images would then be called crops; it matters
  only for images like these, and our evaluation doesn't score them (no weed boxes).
- **Related to ACRE** (same site and sown weeds, see above): a gain on ACRE's beans from
  VCD may be partly site-specific.
- The scale is an estimate, and unknown for the undated images.
- Plants cut by the image border are boxed like the others (the paper gives no rule).

## In the combination

Since 2026-10-30 in `public7` (`configs/combine.yaml`) with CropAndWeed, ACRE, PhenoBench,
OPPD, GROUNDBASED_WEED and LincolnBeet. Leek (`ALLPO`) becomes a class; bean training images
go from 216 to 772, maize from 1,597 to 2,327 (see [twostage.md](../twostage.md#steps),
"Adding Weed-AI datasets").
