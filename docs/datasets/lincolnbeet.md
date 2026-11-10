# LincolnBeet

- Source: Weed-AI's copy,
  https://weed-ai.sydney.edu.au/datasets/dc1e5436-fa3e-4c0f-84e3-3ea875ea9db4
  (`scripts/download_datasets.py`, `configs/lincolnbeet.yaml`; format in
  [weedai.md](weedai.md)). Original: https://github.com/LAR/lincolnbeet_dataset, which links
  a Dropbox zip of PNG images with COCO, VOC and darknet labels and the paper's split.
- Paper: Darbyshire, Salazar-Gomez, Gao, Sklar and Parsons, "Towards practical object
  detection for weed spraying in precision agriculture", Frontiers in Plant Science 14 (2023)
  1183277 (preprint arXiv 2109.11048, Salazar-Gomez first).
- Owner: University of Lincoln, UK (Lincoln Institute of Agri-food Technology, Lincoln
  Centre for Autonomous Systems). Contact in the README: asalazargomez@lincoln.ac.uk.
- Licence: **unclear.** Weed-AI gives CC BY 4.0, but neither the paper nor the GitHub
  repository states a licence (checked 2026-10-30). Weed-AI's licence field is what the
  uploader entered. Before any commercial use, ask the authors.

## Content

From the paper, and measured from the Weed-AI download (2026-10-30):

- **Three sugar beet fields near Lincoln**: "Near 30" and "BBRO", typical commercial fields,
  and a small plot in a walled garden. Filmed in the early growth stages (BBCH 10-17), from
  22 April to 4 June 2021 (dates in the file names): Near 30 on 12 days, BBRO on 5, the
  walled garden on 4.
- **Frames of 4K videos** (3840x2160) from two cameras, 12 MP with a 26 mm and 64 MP with a
  29 mm lens (equivalent focal lengths; phones, probably), about 50 cm above the ground. The
  frames were picked "sufficiently far apart to avoid repetition" and halved to 1920x1080.
  73 images of the walled garden are 1980x1080 and named `_i_` rather than `_v_`: photos,
  probably.
- **Boxes for sugar beets and weeds, the weeds without species.** The paper doesn't say
  whether every plant is boxed. Weed-AI's copy has 4,307 images with 16,026 beets and
  22,208 weeds (the paper: 4,402 images, 16,399 and 22,847).
- File names keep the field, the day and the video:
  `<field>_<field>_<dd>_<mm>_<yyyy>_<v|i>_<video>_<frame>_png.rf.<hash>.jpg`.
- The paper's split is random by image (70/20/10), so frames of one video fall in all three;
  Weed-AI's copy doesn't carry it.

## Conversion choices

- `scripts/convert_weedai.py` (`configs/lincolnbeet.yaml`, listed in `configs/weedai.yaml`),
  reader `weedseg.data.weedai`.
- **Weeds are `1PLAK`**, plants of unknown species: plants for the detector like the beets.
- **Group = field and day** (the file name's first part, videos and photos together): 21
  groups. Videos of one day may cross the same ground. Each field was filmed on several
  days, so val and test share fields, and older or younger plants, with train.
- **mm/px estimated at 0.35**: the paper's own example of a 1.8:1 camera at 500 mm covers
  670 mm across, and the 26 mm lens gives about the same; 670 / 1920 = 0.35. The 29 mm
  lens gives ~0.31.
- **Many plants are under 20 mm**: 11,498 of 38,234 boxes, 10,184 of them weeds (of
  22,208). They become ignore regions. The 66 images left without a larger plant stay, as
  negative images for the detector (`keep_negative_images: true` since 2026-11-06; the
  numbers below are from before, without them).
- **Split 70/15/15 by group, balanced over species, seed 9.** With 21 groups, some large
  (Near 30 on 1 June: 670 images), seed 0 put 36% of the weeds in test; seed 9, the best of
  0-9, keeps both classes within 6 points of the fractions. Test holds BBRO's first two
  days and four Near 30 days, val two Near 30 days and one walled garden day.

| | Train | Val | Test |
|---|---|---|---|
| Images | 2,786 | 516 | 939 |
| Groups (field days) | 12 | 3 | 6 |
| `BEAVA` sugar beet | 9,955 | 1,737 | 3,020 |
| `1PLAK` weed | 8,755 | 1,926 | 1,343 |
| Plants under 20 mm (ignore regions) | 8,329 | 1,347 | 1,626 |

## Known limitations

- **The licence**, see above.
- Weeds have no species, so the per-species report counts them as one label.
- Not stated by the authors whether every plant is boxed; the images we checked had none
  left out.
- Val and test share fields with train (see the groups).
- The scale is an estimate, and the camera height was approximate.

## In the detector's data

LincolnBeet is in `configs/combine.yaml` (`public7`): beets and weeds are boxed, in fields of
another country, with other cameras than the other datasets.
