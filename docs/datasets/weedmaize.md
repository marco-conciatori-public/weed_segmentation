# WeedMaize

- Source: Zenodo record 5106795, https://zenodo.org/records/5106795 (Dataset Ninja has a
  copy). No paper is cited by the record; the description below is Dataset Ninja's and the
  record's.
- Owners: J. M. López Correa, D. Andújar, M. Todeschini, J. Karouta, J. M. Bengochea,
  A. Ribeiro: Universidad Politécnica de Madrid and the Centre for Automation and Robotics
  (CSIC-UPM), Spain. D. Andújar's group also made [GROUNDBASED_WEED](groundbased_weed.md),
  with the same camera in the same region (see [provenance.md](provenance.md)).
- Licence: **CC BY 4.0** (Zenodo record).

## Content

- **Commercial maize fields in Badajoz, Spain** (39°1'14" N, 6°3'41" W), April 2020, maize at
  V1 to V4, no weed control in the 10 days before.
- **Straight down from 1.3 m**, a photo every 2 m along an M-shaped walk, with a Canon
  PowerShot SX540 HS (5184x3888).
- **A box for every plant**, maize and weeds by species, as EPPO codes with a growth stage
  or size class: `ZEAMX`, `ZEAMX_V1`, `ZEAMX_V3`, `ZEAMX_V4`; `CYPRO`, `CYPRO_min`,
  `CYPRO_max`; `ECHCG`, `ECHCG_Ve`, `ECHCG_V1`, `ECHCG_V2`; `SOLNI`, `SOLNI_Vc`, `SOLNI_V1`,
  `SOLNI_V2`; `POROL`; and `NC` and `OE`, which the sources don't explain (plants not
  classified, other species). The description says the overlap of the tiles was used
  "ensuring all plants had labels".
- **Tiles, not photos.** 312 photos were labelled, then cut into 3000x1296 tiles on a grid 400
  px apart (6 per row, 7 rows: tile t at x = 400·((t−1) mod 6), y = 400·((t−1) div 6), found
  2026-11-08 by matching the boxes neighbouring tiles share). The download holds part of
  each photo's 42 tiles, spread over the authors' train, validation and test sets: **tiles of
  one photo are in different sets**, so that split leaks.
- Some plants are boxed twice, a pixel or two apart.
- **Cyperus is boxed twice on purpose**: `CYPRO_max` around the whole plant, `CYPRO_min`
  around its centre, inside it (seen in the previews 2026-11-08; GROUNDBASED_WEED's paper
  describes the same two conventions). On the validation tiles, 90% of the 2,385 `CYPRO_min`
  boxes lie inside a `CYPRO_max` (IoA >= 0.9), 4% partly, 6% inside none; 22% of the 2,764
  `CYPRO_max` boxes have no `CYPRO_min` inside. Plain `CYPRO` is a single box. Other weeds inside
  a `CYPRO_max` box are separate plants under its sparse leaves.
- The previews of 6 validation tiles showed every plant boxed, seedlings included (`NC`,
  `SOLNI_Vc`, `ECHCG_Ve`).

## Conversion choices

- `scripts/convert_weedmaize.py` (`configs/weedmaize.yaml`), shared pipeline
  `weedseg.data.convert`, reader `weedseg.data.weedmaize`.
- **A spread-out set of tiles per photo** (`max_tile_overlap: 0.35`): greedily, the tile
  adding the most uncovered ground first, as long as it shares at most 35% of its area with
  each tile kept. From a full grid that keeps 6 tiles (columns at 0 and 2000 px, rows at 0,
  1200 and 2400 px), covering the photo with a strip seen twice. In a full grid, a point of
  the photo is in up to 24 tiles.
- Tiles, not reassembled photos: at 1088 px, a 1296 px tile is seen at about 0.43 mm/px,
  close to CropAndWeed; a whole photo would shrink to about 1.2 mm/px.
- **Duplicate boxes dropped** (`duplicate_iou: 0.9`, any label).
- **One box per Cyperus** (`centres: {CYPRO_min: CYPRO_max}`): a `CYPRO_min` at least half
  inside a `CYPRO_max` is dropped; one inside none becomes an ignore region (`centre_only`),
  since the plant's extent wasn't drawn.
- All the classes map to their EPPO species (`ZEAMX` crop); `NC` and `OE` are `1PLAK`.
- **mm/px estimated at 0.36**: the camera at its widest zoom (4.3 mm, 1.19 µm pixels) from
  1.3 m.
- **Groups are blocks of 10 consecutive photo numbers** (`group_block`); the authors' sets
  are not used.

Converted 2026-11-08: the download holds 7,862 tiles of **156 photos** (the description
speaks of 312 labelled photos); 936 tiles are kept, exactly 6 per photo. Plants of at least
20 mm:

| | Train | Val | Test |
|---|---|---|---|
| Tiles (groups) | 666 (19) | 162 (3) | 108 (3) |
| `ZEAMX` maize | 3,187 | 884 | 508 |
| `CYPRO` | 2,386 | 297 | 424 |
| `SOLNI` | 510 | 67 | 68 |
| `ECHCG` | 389 | 30 | 144 |
| `1PLAK` (`NC`, `OE`) | 217 | 29 | 44 |
| Ignore regions: under 20 mm / centre only | 2,682 / 60 | 387 / 13 | 462 / 4 |

Median longest side at 0.36 mm/px: maize 95 mm at V1, 197 mm at V3, 270 mm at V4; weeds at
cotyledons about 22 mm, which fits the scale. `POROL` is in only 5 boxes of all the tiles,
none kept as a plant.

## Known limitations

- **Tiles, cut after labelling**: a few plants cut by a tile's edge have no box (maize leaf
  tips at a tile's top or bottom, seen in the previews).
- One region, one season, one camera, and the same group, camera and region as
  GROUNDBASED_WEED.
- mm/px is an estimate (EXIF stripped from the tiles).
