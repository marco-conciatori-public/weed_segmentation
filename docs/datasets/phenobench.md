# PhenoBench

- Source: https://www.phenobench.org, version 1.1.0 (December 2023), one 7.6 GB zip
  (`scripts/download_datasets.py`, `configs/phenobench.yaml`). Paper: Weyler et al.,
  "PhenoBench: A Large Dataset and Benchmarks for Semantic Image Interpretation in the
  Agricultural Domain", IEEE TPAMI 2024 (arXiv 2306.04557).
- Owner: University of Bonn.
- Licence: **CC BY-NC-SA 4.0**, per the README.MD in the download (the website and the
  Hugging Face copy said CC BY-SA 4.0). Non-commercial and share-alike: like CropAndWeed,
  keep it out of any model meant for commercial use.

## Content

- Drone images of one sugar beet field near Bonn (Campus Klein-Altendorf), on 15 May,
  26 May and 5 June 2020: PhaseOne iXM-100 camera with an 80 mm lens at about 21 m, a ground
  resolution of 1 mm/px (paper).
- The 1024×1024 images are overlapping windows of the 11,664×8,750 px photos. Labelled:
  1,407 train images from 21 photos and 772 val images from 12 photos. The 693 test images
  (which include a field from 2021) have no public labels.
- The split is spatial: 4 crop rows for train, 2 for val, 2 for test.
- Per pixel: soil, crop, weed, partial crop, partial weed ("partial" = less than half of
  the plant visible, cut by the image border or covered by a neighbour), and an instance
  mask for every plant. Leaf instances and visibility maps are not used.
- Weeds are one class: the paper names six species but doesn't label them.
- The same plants appear on several dates (496 on all three, per the paper) and in
  overlapping images.

| | Train | Val |
|---|---|---|
| Sugar beet (*Beta vulgaris*, `BEAVA`) | 8,938 (8,679 at least 20 mm) | 4,967 (4,841) |
| Weed, species unknown (`1PLAK`) | 7,512 (5,967) | 3,601 (3,090) |
| Partly visible crops and weeds (ignore regions) | 3,566 | 1,840 |

## Conversion choices

- `scripts/convert_phenobench.py` (`configs/phenobench.yaml`), shared pipeline
  `weedseg.data.convert`.
- **PhenoBench's own split is kept**, since it separates crop rows: its train and val images
  are our train and val splits, and our test split stays empty. Its val is not divided
  further: overlapping images of the same two rows would put the same plants in our val
  and test. The group (recorded, not used for splitting) is the drone photo.
- Sugar beet is `BEAVA`; weeds are `1PLAK`, plants of unknown species: ignore regions in the
  species files, plants for the two-stage detector ([data-format.md](../data-format.md)).
  Their label map row says `none_of` `BEAVA`, so the classifier learns them as "anything but
  sugar beet" ([partial labels](../data-format.md#partial-labels)). Without that, every
  PhenoBench plant the classifier saw was a beet, and it called all PhenoBench weeds sugar
  beet (run 2, 2026-10-23).
- **Partly visible plants become ignore regions** (`ignore_partial_plants`), keeping their
  code. A plant cut by the image border is at least half visible in an overlapping image:
  the paper chose the overlap for that.
- mm/px is the paper's 1.0 (`mm_per_px_estimated: false`).
- Every plant pixel belongs to an instance and every instance has a single label (checked
  2026-10-21), so no ignore regions for unassigned vegetation are needed.

## In the combination

- The class list doesn't change: sugar beet is already a class (from CropAndWeed). A trial
  combination with CropAndWeed and ACRE (2026-10-21) gave the detector 14,646 more train
  plants (+26%) and the classifier 8,679 more sugar beet crops.
- Its weeds reach only the detector.
- Since 2026-10-22 in `configs/combine.yaml` with CropAndWeed, ACRE and OPPD, and since
  2026-10-26 with GROUNDBASED_WEED too:
  `data/processed/cropandweed+acre+phenobench+oppd+groundbased_weed`, the data of the
  two-stage configs.

## Known limitations

- **Little variety**: one field, one season, three dates, 33 photos, the same plants
  repeated across dates and overlapping images.
- One crop only, and weeds without species.
- Scored on val only: it has nothing in our test split.
