# CornWeed

- Source: Zenodo record 7961764, https://zenodo.org/records/7961764 (also on Weed-AI). Paper:
  Naeem, Manss, Scholz, Koenig, Igelbrink, Ruckelshausen, "AI-based Maize and Weeds detection
  on the edge with CornWeed Dataset", FedCSIS 2023 (Annals of Computer Science and
  Information Systems 35).
- Owners: German Research Center for Artificial Intelligence (DFKI) and Hochschule Osnabrück
  (University of Applied Sciences), Germany.
- Licence: **CC BY 4.0** (Zenodo record).

## Content

From the paper and the download (checked 2026-11-08):

- **One field**, "Langsenkamp" (the file names; the paper: "the same field such that the same
  soil conditions and the same types of weeds persist"), maize at early growth stages, filmed
  over three weeks in July-August 2021, by daylight, cloudy and sunny.
- **Camera straight down**: an Intel RealSense D435i "looking downwards", on two machines: an
  implement on a tractor hoe (Sensorbox 1, 1280x720) and the BoniRob research platform
  (Sensorbox 2, 640x480). The paper puts both cameras at 0.5 m, so that "the geometric size
  of each obtained image spans a distance of 0.68 m along the driving direction"; the images
  don't fit that (see the scale below).
- The rows run top to bottom in the images. The tractor images show two maize rows (in 1,729
  of 1,849 images) and the hoe's gauge wheel at the left edge; the BoniRob images show one
  row. The BoniRob plots have a fine herringbone print on the soil that the tractor's don't.
- **Boxes for every maize plant and every weed**, two classes (`Maize`, `Weeds`, no species).
  Labelled in CVAT by annotators and checked by a second person; proposals of an interim
  detector were corrected by "adding not detected weeds and maize".
- **Labelled runs** (file name `<field>_<yymmdd>[_bonirob]_color_<frame>.png`):

| Run | Machine | Size | Labelled images |
|---|---|---|---|
| 2021-07-22 | BoniRob | 640x480 | 1,054 |
| 2021-07-27 | tractor | 1280x720 | 1,849 |
| 2021-07-27 | BoniRob | 640x480 | 671 |

  3,574 labelled images with 257,740 weed and 23,985 maize boxes over both JSON files: about
  72 weeds per image, many small. A fourth run (2021-08-02, BoniRob) is only in the 4,981
  unlabelled `testing/` images, which are not used.
- The frames are consecutive video frames, so neighbouring ones show the same plants. The
  authors' train/valid split is by frame.

## Conversion choices

- `scripts/convert_cornweed.py` (`configs/cornweed.yaml`), shared pipeline
  `weedseg.data.convert`, reader `weedseg.data.cornweed`. Both annotation files are read; the
  split is ours.
- `Maize` is `ZEAMX` (crop), `Weeds` is `1PLAK` (plants of unknown species).
- **Groups are blocks of 50 consecutive frames of one run** (`group_block`), never single
  frames.
- **mm/px per camera setting** (`mm_per_px_by_width`), estimates made 2026-11-09:
  - **1280x720 (tractor): 1.07 mm/px.** The two maize rows are a median 699 px apart (p25-p75
    672-725 px); at the usual 75 cm between maize rows that is 1.07 mm/px. The paper's
    geometry (0.5 m, 69 degrees) gives 0.68 m over the 1280 px, 0.53 mm/px, which would put the
    rows 37 cm apart, make the tyre print in the inter-row (280 px) 15 cm wide and the longest
    leaves of maize with 5-6 leaves (about 310 px) 16 cm long. At 1.07 these are 30 cm and
    33 cm, and the in-row spacing (median gap between maize centres, 153 px) 16 cm, all
    usual for maize. The camera is then about 1 m up.
  - **640x480 (BoniRob): 1.3 mm/px.** Maize of the same day (27 July) is 1.14x smaller in px
    (median longest side of the boxes clear of the edges, 279 vs 244 px) and its in-row gaps
    1.23x shorter (153 vs 124 px, the 22 July run's cleaner gaps); about 1.2x the tractor's
    mm/px. The weeds of that day point to 1.5x (median 28.1 vs 18.4 px), as does the camera
    (640x480 is a crop of the sensor at 1.5x the pixel size of 1280x720) if both were at the
    same height; the maize is used because both machines drove in the same field sown once,
    while the weeds depend on each plot's history (the plots differ, see the soil above).
  - The scale only drives the 20 mm size filter here: at 0.53/0.81 a quarter of the weeds were
    kept as plants, at 1.07/1.3 three quarters.
- **Every second frame** (`frame_stride: 2`, frame numbers divisible by 2). The labelled
  images are consecutive video frames, and the image moves about half its height from one to
  the next (median 328 of 720 px on the tractor, 244 and 272 of 480 px on the BoniRob runs,
  from boxes of the same size shifted alike in both frames; overlap 43-56%), so neighbours
  share half their plants. Every second frame shows the ground about once; no two
  consecutive frames repeat the same boxes (the vehicles never stood still).

Converted 2026-11-09: 1,789 of the 3,574 frames, plants of at least 20 mm:

| | Train | Val | Test |
|---|---|---|---|
| Images (groups) | 1,235 (54) | 269 (11) | 285 (14) |
| of which tractor / BoniRob 22 July / BoniRob 27 July | 696 / 377 / 162 | 50 / 100 / 119 | 180 / 50 / 55 |
| `ZEAMX` maize | 8,551 | 1,356 | 2,080 |
| `1PLAK` weeds | 68,990 | 11,385 | 16,682 |
| Ignore regions (under 20 mm) | 22,024 | 6,066 | 3,652 |

Every image has plants. The split is balanced over the two classes, not over the runs: the
validation set has only 50 tractor images (2 groups). The previews of 6 validation images
(2026-11-09, before the thinning) show every plant boxed, seedlings included; a few weed
boxes enclose a clump of several weeds.

## Known limitations

- **The scale is an estimate** that contradicts the paper (see above); the 75 cm row
  distance is assumed, not given.
- **One field, one season, two days labelled**: little variety of soil, light and weeds; the
  tractor run alone is half the images and two thirds of the plants.
- **Many plants per image** (about 60 kept, mostly weeds of 2-4 cm): even with every second
  frame, CornWeed's 109k plant boxes are 45% of those of the five datasets of `public5`
  together (243k; all frames: 218k). In training it weighs by its images, though (1,235 in
  train), since the detector samples a fixed number of anchors and proposals per image. In
  `public6` since 2026-11-09 (Marco; `public7` since DeepSeedling joined); if the other datasets' val numbers drop, keep fewer
  frames (`frame_stride`).
- Weeds without species (`1PLAK`): fine for the plant detector.
