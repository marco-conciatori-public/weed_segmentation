# DeepSeedling

- Source: figshare private link https://figshare.com/s/616956f8633c17ceae9b (from the GitHub
  repo UGA-BSAIL/deepseedling). Paper: Jiang, Li, Paterson, Robertson, "DeepSeedling: deep
  convolutional network and Kalman filter for plant seedling detection and counting in the
  field", Plant Methods 15 (2019) 141 (open on PMC as PMC6874826).
- Owner: Bio-Sensing and Instrumentation Laboratory, University of Georgia (with Texas A&M
  University for the 2015 Texas data).
- Licence: **GPL 3.0+** (shown on the figshare page; seen by Marco 2026-11-10; the GitHub
  repo has no licence file). Commercial use is allowed, but it is copyleft: whatever is
  distributed "based on" the data must be GPL too, and whether a model trained on it counts
  is legally open. Today's detector is non-commercial anyway (CropAndWeed, PhenoBench and
  OPPD); for a commercial model, drop it from `configs/combine.yaml` or get legal advice.

## Content

From the paper (Table 4) and the download (checked 2026-11-10):

| Set | Site | Date (days after planting) | Camera | Carried | Seed spacing | Labelled frames |
|---|---|---|---|---|---|---|
| TAMU2015 | Corpus Christi, Texas, USA; 35 commercial cultivars, plots 10.67 m | 12 April 2015 (11) | Samsung Galaxy Note3 (a phone), video 1920x1080 at 30 fps | by hand | 8 cm | 1,821 (`Row20`) |
| UGA2015 | Watkinsville, Georgia, USA; breeding lines, plots 3.05 m | 15 June 2015 (11) | Panasonic DMC-G6, 60 fps; frames 1401x1080 | cart | 10 cm | 2,389 frames in 4 folders (`102`, `104`, `116`, `117`), 1,960 labelled |
| UGA2018 | Watkinsville, Georgia, USA; 4 commercial cultivars, plots 1.5 m | 13 June 2018 (7) | Fujifilm X-A10, 1920x1080 at 30 fps | cart | 15 cm | 1,531 frames of six videos in `Images`, 1,511 labelled |

- "The camera was placed/held at approximately 0.5 m above the ground"; the frames look
  close to **straight down** (checked on samples). TAMU2015 is a hand-held phone close to the
  plants; a hand is in some frames, pointing at or touching seedlings. UGA2018 has motion
  blur. The camera's angle isn't stated in the paper.
- **Boxes for cotton seedlings (`Plant`) and weeds (`Weed`, no species)**, labelled in VoTT
  1.5. "Very small-sized weed objects (less than 30 × 30 pixels) were not labeled": the 30 px
  were on VoTT's screen, which showed UGA2015's frames at 826 px wide for 1,401, so there
  seedlings under ~50 px (~17 mm) have no box (the smallest boxed weed is 37 px, 5th
  percentile 49 px); one sample frame has ~15 of them next to 12 boxes. TAMU2015 and UGA2018
  samples showed only tiny sprouts unboxed. **Worse, found after converting:** in UGA2015's
  weedy stretches many weeds over that size have no box either (folder `104` around frame
  3751: ~25 weeds in view, 0-5 boxed; so UGA2015 is left out, see Decision). A clump of
  purslane can get one box. One box (UGA2015 `104`) is tagged both `Plant` and `Weed`: label
  `Plant+Weed`, a plant of unknown species (`1PLAK`). UGA2018 has no weeds (pre-emergence
  herbicide; one `Weed` box); TAMU2015's weeds are mostly grasses and purslane, UGA2015's
  broadleaves.
- Counts over the label files (boxes): TAMU2015 `Row20` 18,977 seedlings, 726 weeds; UGA2015
  6,896 seedlings, 962 weeds; UGA2018 5,883 seedlings, 1 weed.
- The paper contrast-equalised the frames (CLAHE) before labelling; the released frames look
  like the cameras' own (UGA2015's are dark), so the boxes are on unprocessed images.
- **The frames come from videos**, at 5-6 per second while moving 0.6-0.75 m/s, so
  neighbouring frames share most of their plants. Measured by matching consecutive frames
  (phase correlation, 2026-11-10): UGA2015 moves ~250 px of its 1,080 px height per frame,
  UGA2018 ~500 px of its 1,920 px width (a weak match: blur, sandy soil), and the hand-held
  TAMU2015 10-50 px where the hand is in view; over its ~75 m of row (7 plots) in 1,821
  frames, ~100 px on average.
- VoTT label files (`<folder>.json`): frame i is the i-th image of the folder as Windows lists
  it (by name, ignoring case: `1001.jpg` comes before `121.jpg`, checked on samples where the
  boxes fit only that order); box coordinates are on VoTT's screen (each box carries the
  screen's `width` and `height`). Frames missing from the file were not labelled (UGA2015
  folders `104` and `117` have ~200 each). TAMU2015 also holds `Row10.mp4` (0.8 GB), labelled
  on the video file (`Row10.mp4.json`, 392 frames): not used.
- The authors' split is by frame (80/10/10).

## Conversion choices

- `scripts/convert_deepseedling.py` (`configs/deepseedling.yaml`), shared pipeline
  `weedseg.data.convert`, reader `weedseg.data.deepseedling`. Download with
  `scripts/download_datasets.py`: the figshare page refuses scripts, but its download links
  work (each redirects to a 10-second storage link, so a browser can't resume an interrupted
  download, while the downloader asks again for every piece).
- `Plant` is cotton, `GOSHI` (*Gossypium hirsutum*, upland cotton: the paper says only
  "cotton", but the cultivars are commercial ones of Texas and Georgia), crop; `Weed` is
  `1PLAK` (plants of unknown species). GOSHI was added to `configs/eppo_taxa.csv` on its own
  (rebuilding the table from this branch's label maps would also drop 11 monocot orders it
  holds).
- Only frames with boxes are used (`keep_negative_images: false`): a VoTT frame without boxes
  may have been skipped rather than empty.
- **mm/px per set** (`sets`, estimates made 2026-11-10; the paper gives only "about 0.5 m"):
  - **UGA2015: 0.35** (left out, see Decision). 3.5 labelled seedlings per frame along a row that crosses the 1,080 px
    height: one per 308 px. At 10 cm seed spacing and 85-100% emergence, 0.32-0.38 mm/px.
    The G6 (Micro Four Thirds) at 0.5 m with its 14 mm kit lens at the widest gives
    0.32 mm/px over 1,920 px (the lens isn't stated).
  - **UGA2018: 0.35.** 3.9 seedlings per frame along a row across the 1,920 px width: one
    per 492 px; at 15 cm, 0.30-0.36 mm/px. The X-A10 (APS-C) at 0.5 m with its 16 mm kit lens
    at the widest gives 0.38.
  - **TAMU2015: 0.4.** 10.4 seedlings per frame along a row across the width: one per 184 px;
    at 8 cm, 0.43-0.51 mm/px, an upper bound (frames over the alleys between plots hold
    fewer plants). The Note3 (~60 degrees wide in video) at 0.5 m gives 0.30. Between the two.
  - The seedlings then measure (median longest side of the boxes clear of the edges) 42 mm
    (TAMU2015, 104 px), 35 mm (UGA2015, 100 px) and 49 mm (UGA2018, 141 px).
- **Frames thinned per set** (`frame_stride`, in time order within each video): 4 for
  UGA2018 (the ground about once), 10 for TAMU2015 (each plant about twice on
  average; where the phone stands still, neighbours are near copies).
- **Groups are blocks of consecutive labelled frames of one video** (`group_block`, before
  thinning): 60 in UGA2018 (a plant stays in view ~4 frames), 180 in TAMU2015
  (~19). Videos are `<set>/<folder>` and, in UGA2018, the file-name prefix (`ENGR_1`,
  `NAM3_2`, ...).

## Converted (2026-11-10)

`data/processed/deepseedling/species_min20mm/`, TAMU2015 and UGA2018 only (UGA2015 left out,
see Decision): 563 images (frames with boxes, thinned).

| | TAMU2015 | UGA2018 |
|---|---|---|
| Labelled frames with boxes | 1,820 | 1,511 |
| Kept (`frame_stride`) | 182 | 381 |

Splits: train 386 images from 26 groups (2,459 plants), val 86 from 6 (571), test 91 from 6
(395); 3,354 cotton, 71 weeds (`1PLAK`, nearly all TAMU2015's), 29 plants under 20 mm as
ignore regions. The first conversion, with UGA2015 (473 more images), had 1,036 images, 5,049
cotton and 287 weeds. Previews of 14 frames, two or three per folder (`preview_annotations.py`,
UGA2015's folders included), confirm the frame mapping in every folder.

## Decision

Checked strictly in tier 2 of the detector's dataset survey (2026-11-10). **Trained on
TAMU2015 and UGA2018; UGA2015 left out** (Marco, 2026-11-10, option 2). They bring a new crop,
country and cameras, a phone among them. The cost of leaving UGA2015 out: most of
DeepSeedling's broadleaf weeds (71 weeds remain, mostly TAMU2015's grasses and purslane).
In `public7` since 2026-11-10.

How it was decided: option 1 (all three sets trained on; the others were UGA2015 as test
only, or all of it as test only) was chosen first, when only UGA2015's seedlings under ~17 mm
seemed unboxed. The previews after converting then showed weedy stretches of UGA2015 where
weeds as large as the boxed ones have no box (folder `104`, frames ~3600-3900: frame 3851 has
5 weeds boxed of ~25 in view). A count of green patches over 20 mm with no box (excess-green
threshold, a lower bound: it misses plants in shadow) gives per image: TAMU2015 0.18 (15% of
images with one or more, none with 3), UGA2015 0.63 (29%; 7% with 3 or more, up to 13),
UGA2018 0.16 (14%, none with 3). TAMU2015's and UGA2018's are mostly plants cut by the image
edge; UGA2015's are unboxed weeds, which the detector would learn as soil. UGA2015 isn't kept
as a test set either: its unboxed weeds would count as false detections. Its zip is no longer
downloaded; the reader still handles it (and its one box tagged both ways, `Plant+Weed`).
