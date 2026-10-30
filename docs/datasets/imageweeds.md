# ImageWeeds

- Source: Weed-AI's copy, seven uploads, one per crop (`configs/imageweeds.yaml` lists them;
  e.g. flax: https://weed-ai.sydney.edu.au/datasets/ab3da4ea-4cb4-407a-a007-7052ed3a7850).
  `scripts/download_datasets.py`; format in [weedai.md](weedai.md). Original: Mendeley Data
  "ImageWeeds", doi 10.17632/8kjcztbjz2.2, which also holds close-up images of single weeds
  (not used).
- Paper: Rai et al., "ImageWeeds: an image dataset consisting of weeds in multiple formats
  to advance computer vision algorithms for real-time weed identification and spot spraying
  application", Data in Brief 51 (2023) 109691 (open on PMC as PMC10618417).
- Owner: North Dakota State University (USDA and NIFA funding).
- Licence: **CC BY 4.0** (Mendeley Data record, and Weed-AI).

## Content

From the paper, and measured from the Weed-AI download (2026-10-30):

- **Drone photos of a weed trial** at three North Dakota sites: the Agronomy Seed Farm in
  Casselton (May-June 2021), the Carrington Research Extension Center (July-August 2021) and
  Grand Farm (August-September 2022). DJI Phantom 4 Pro V2 at ~3.7 m, 5472x3648 (also
  4864x3648 and 5472x3078). The paper gives no ground resolution and no camera angle;
  Weed-AI says 70 degrees, but the photos look straight down (the drone's shadow is below
  it).
- **The weeds are potted plants placed in the field.** Each plot has crop rows side by side
  with a line of black square pots in the centre, sunk into the soil, each with one or two
  weeds of the five species. Only the potted weeds are boxed: the crop and the field's own
  weeds around them are not.
- Five species, boxes only: common ragweed (*Ambrosia artemisiifolia*, `AMBEL`), kochia
  (*Bassia scoparia*, `KCHSC`), waterhemp (*Amaranthus tuberculatus*, `AMATU`), horseweed
  (*Erigeron canadensis*, `ERICA`) and redroot pigweed (*Amaranthus retroflexus*, `AMARE`).
- Weed-AI split the photos by crop into seven uploads (sugar beet, brassica, field pea, flax,
  soybean, lentil, maize): 459 photos, 2,016 boxes. The paper counts ~550 photos and 2,390
  boxes. The crops are not named in the paper; nothing tells which site a photo comes from.
- The file names `UASs_<n>` number all uploads in one sequence (1-1441), and neighbouring
  numbers fall in different crops: the drone flew over the plots in turn, back and forth.
  Feature matching found 53 pairs of photos up to 25 numbers apart that show the same ground
  (e.g. 6 and 19, 22 and 41).
- **54 photos are there twice**, byte for byte, under two numbers 313 apart (1000-1061 and
  1313-1374) or 537 apart (117, 123, 124 and 654, 660, 661), always in the same crop upload.
  The two copies were labelled separately and differ (a plant more, two plants in one box).

## Conversion choices

- `scripts/convert_weedai.py` (`configs/imageweeds.yaml`, listed in `configs/weedai.yaml`),
  reader `weedseg.data.weedai`. **Box only**: the classifier learns from these plants,
  Mask R-CNN doesn't see them.
- **Copies left out** (`drop_duplicate_images`): of each pair of identical photos the lower
  number is kept, so 405 photos and 1,862 weeds remain.
- **Group = a block of 50 consecutive photo numbers**, across the crop uploads: 18 groups.
  Blocks of 20 would split 11 of the 53 overlapping pairs, blocks of 50 only one (1299 and
  1300). Pairs more than 25 numbers apart were not checked.
- **mm/px estimated at 1.0**: 13.2 mm sensor over 5472 px, 8.8 mm lens, 3.7 m. The other
  image modes keep the pixel size.
- **Split 70/15/15 by group, balanced over species, seed 5.** With 18 groups, seed 0 put 26%
  of the redroot pigweeds in test; seed 5, the best of 0-9, keeps every species within 6
  points of the fractions. Every box is at least 20 mm.

| | Train | Val | Test |
|---|---|---|---|
| Photos | 260 | 76 | 69 |
| Groups (blocks of 50 numbers) | 11 | 3 | 4 |
| `AMBEL` common ragweed | 344 | 93 | 78 |
| `KCHSC` kochia | 328 | 55 | 102 |
| `AMATU` waterhemp | 220 | 44 | 68 |
| `ERICA` horseweed | 187 | 43 | 51 |
| `AMARE` redroot pigweed | 178 | 43 | 28 |

## Known limitations

- **Potted weeds.** The classifier may learn the pot (black rim, the hole around it) as part
  of these five species, as it may have learned OPPD's trays: species learned mostly from
  OPPD transferred badly to CropAndWeed's fields. Four of the five species come only from
  this dataset. Redroot pigweed is the check: it is also in CropAndWeed, so its accuracy
  there with and without ImageWeeds shows whether the pots help or hurt.
- One trial design at three sites; the site of each photo is unknown, so val and test
  share sites with train.
- Small: 260 training photos. The four new species have 187-344 training plants each, in
  92-141 photos per species (2026-10-30), and neighbouring photos can show the same pots.
- The crops and the field's own weeds are visible but unlabelled.
