# Sugar Beets 2016 (part 1, an extra test set)

- Source: https://www.ipb.uni-bonn.de/data/sugarbeets2016/ (files under
  https://www.ipb.uni-bonn.de/datasets_IJRR2017/). Paper: Chebrolu, Lottes, Schaefer,
  Winterhalter, Burgard, Stachniss, "Agricultural robot dataset for plant classification,
  localization and mapping on sugar beet fields", IJRR 36(10), 2017.
- Owners: University of Bonn (and University of Freiburg), Germany.
- Licence: **CC BY-SA 4.0** (dataset page).

## Content

- **One sugar beet field at Campus Klein-Altendorf near Bonn**, spring 2016, recorded about
  three times a week from emergence on: PhenoBench's site, group and robot (see
  [provenance.md](provenance.md)).
- **BoniRob, camera straight down**: a JAI camera (1296x966) "at a height of around 85 cm
  above soil", 8 mm lens; the paper gives "approximately 3 px/mm" (0.33 mm/px) and a field of
  view of 24 x 31 cm, which don't agree (0.24 mm/px) and neither agrees with the optics
  (1/3" sensor, 0.40 mm/px). The images are dark.
- **Labels are masks per pixel, not plants**: `annotations/dlp/iMapCleaned` holds 0 (soil),
  10000 (sugar beet), 2 and 20100 (weed, both red in the colour masks); there are no plant IDs.
  Dataset Ninja's "instances" are connected patches of these masks.
- **The labelled images come as a RAR archive in 11 parts, 22 GB** (no checksum). Part 1
  (2 GB) holds the day 2016-05-23: 8 runs, 1,931 RGB images, 2,148 index maps.

## Why only part 1, and only for testing

Checked 2026-11-09 on part 1. Boxes made from the masks' connected patches are right for a
beet or weed on its own, but:

- **touching beets become one box**: 55% of the images have a beet patch over twice the
  day's median beet patch (e.g. `bonirob_2016-05-23-10-47-22_2_frame200`: two or three beets
  in one box);
- **plants in pieces become several boxes**: grass leaves that don't touch in the image; 57%
  of the weed patches lie within 10 px of another weed patch;
- **some plants have no mask** (e.g. a grass weed in `bonirob_2016-05-23-10-37-10_0_frame290`,
  tiny seedlings).

It is also PhenoBench's site and robot. Marco's decision (2026-11-09): not for training; part 1
is kept as a second test of how the detector and imperfect labels behave in such situations,
next to the regular test sets that measure the detector. The other 10 parts are not fetched.

## Conversion

- `scripts/convert_sugarbeets2016.py` (`configs/sugarbeets2016.yaml`), shared pipeline
  `weedseg.data.convert`, reader `weedseg.data.sugarbeets2016`. The script first unpacks the
  RAR part with 7-Zip (`seven_zip`; part 1 alone ends inside a file of part 2, which is left
  out).
- Each 8-connected patch of the beet or weed mask is one box (`min_patch_px: 50` drops
  specks), with its pixel count in the annotation field `patch_px` (a beet patch far bigger
  than the others is likely several plants). `beet` is `BEAVA` (crop), `weed` is `1PLAK`.
- **All images go to `test.json`**, no train or val. Groups are the runs.
- mm/px: 0.33, the paper's figure.

Converted 2026-11-09: 1,930 images (8 runs; 8 without plants), 3,997 beets and 2,978 weeds of
at least 20 mm; ignore regions (under 20 mm): 809 beet and 23,985 weed patches.
