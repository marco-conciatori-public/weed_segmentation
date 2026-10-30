# Weed-AI

[Weed-AI](https://weed-ai.sydney.edu.au/datasets) is the University of Sydney's open
repository of weed image datasets. Contributors upload their data, which Weed-AI turns into
one format (WeedCOCO) with metadata on how the images were taken. Each upload has a page,
`https://weed-ai.sydney.edu.au/datasets/<upload id>`, showing the camera, height, lens,
location, growth stages, licence and the boxes per class. 41 uploads were listed on
2026-10-30.

We use three of them: [LincolnBeet](lincolnbeet.md), [ImageWeeds](imageweeds.md) and
[VCD](vcd.md). Their places and dates are compared with the other datasets in
[provenance.md](provenance.md).

## Getting the data

- The dataset page is built from `api/upload_info/<upload id>` (JSON), and the list of
  uploads from `api/upload_list/`. Both can be read by a script.
- The download, `code/download/<upload id>.zip`, is the current version of the upload:
  `weedcoco.json` plus `images/`. Byte ranges work, ~2 MB/s from our network on 2026-10-30.
  Weed-AI's version links (`<upload id>.v<n>.zip`) don't follow the version numbers its
  metadata gives (lentils "v4" is a 404, v1 works), so the configs use the current version,
  and the `size` in each config pins it: a changed upload fails the size check.
- A dataset split over several uploads (ImageWeeds and VCD: one per crop) is unpacked one
  upload per folder (`extract_to`), since every zip has the same layout.

## WeedCOCO

COCO with Weed-AI's additions:

- Categories are `<role>: <latin name>` in lower case, e.g. `crop: beta vulgaris`,
  `weed: amaranthus palmeri`, or `weed: UNSPECIFIED` for weeds without a species. Some
  add a growth stage or a plant part, e.g. `weed: raphanus raphanistrum (BBCH14-15)`,
  `crop: zea mays (stem)`. The label maps take these names as source labels.
- `agcontexts`: camera make, lens, focal length, height, angle, field of view, location,
  BBCH range, lighting, soil, as the uploader filled them in. The angle field isn't
  reliable: several top-down datasets give 0 (probably "unspecified"), and ImageWeeds's 70
  degrees is not in its paper.
- Boxes (`bbox`); masks (`segmentation`) in only one upload of the 41.
- Uploads may hold the same image twice: ImageWeeds has 54 photos under two numbers, each
  copy labelled separately.
- No dates or sites per image. The original file names often still tell them: uploads
  made through Roboflow keep them as `<original stem>_<ext>.rf.<hash>.<ext>`. Some older
  uploads have hashed names and no EXIF, so they can't be split by session.

## Converter

`scripts/convert_weedai.py` converts the datasets listed in `configs/weedai.yaml`, each
described by its own config (`configs/lincolnbeet.yaml`, ...), with the shared pipeline
`weedseg.data.convert` and the reader `weedseg.data.weedai`. All Weed-AI images are
converted as box only ([data-format.md](../data-format.md#datasets-labelled-with-boxes-only)):
they reach the species classifier, not Mask R-CNN.

A dataset config has, besides its `download` section:

| Key | |
|---|---|
| `name`, `source`, `licence` | written into the converted files' `info` |
| `convert.parts` | the uploads (folders under `raw_dir`) that make up the dataset |
| `convert.drop_labels` | boxes that aren't plants, left out (VCD's stem boxes) |
| `convert.drop_duplicate_images` | leave out images whose file is byte for byte an earlier one's (ImageWeeds has 54) |
| `convert.group_pattern` | a regular expression on the original file name; its first capture group is the image's group. Matched from the start. |
| `convert.group_block` | when set, the capture is a number and the group a block of that many consecutive numbers (ImageWeeds) |
| `convert.mm_per_px_estimate` or `convert.footprint_mm_estimate` | the scale, the same for every image, or as the ground length of each image's longer side (for images resized to different sizes, VCD). One of the two, the other null. |

The image field `source_name` keeps the original file stem, `part` the upload.

### Adding another Weed-AI dataset

1. Read its page and its paper. Check the licence on the original release too: Weed-AI's
   licence field is what the uploader entered (LincolnBeet: CC BY 4.0 on Weed-AI, no licence
   in the original).
2. Fill its row in [provenance.md](provenance.md) and compare it with the others.
3. Write `configs/<dataset>.yaml` (download and convert sections, as in the existing ones)
   and `configs/label_maps/<dataset>.csv` (source label = the WeedCOCO category name), then
   run `scripts/build_eppo_taxa.py` for new EPPO codes.
4. Add the config to `configs/download.yaml` and `configs/weedai.yaml`, download, convert.
5. Add the paper to `PAPERS` in `weedseg.deploy.export`, and a dataset doc here.

## The survey of 2026-10-30

All 41 uploads were checked against what the two-stage pipeline can use. Nearly all are
boxes only (classifier only), and most box only the weeds, not the crop.

**Used:** LincolnBeet, ImageWeeds (all seven crop uploads), VCD (beans, maize, leeks).

**Possible later:**

- CornWeed (Osnabrück, Germany): 3,574 images, maize plus 258k weeds without species.
  RealSense camera at 50 cm, 640x480 and 1280x720.
- The wild radish uploads (*Raphanus raphanistrum*): RadishWheatDataset (top-down only), Wild
  radish in wheat, Danish Winter Radish, Danish pasture weeds. About 1,400 boxes, nearly all
  from one farm (Cobbity, NSW).
- Danish thistles (*Cirsium vulgare*, 482 boxes, phone).
- Broadleaf Weeds in Common Couch (*Taraxacum officinale*, 533 boxes, phone, Perth).

**Not used:**

| Upload(s) | Why |
|---|---|
| 7 GRDC wheat/chickpea sets (Narrabri, Cobbity) | the weeds are seedlings under the wheat canopy and the wheat isn't boxed, so the crops would be mostly wheat; hashed names |
| RadishWheatDataset 30/45/60 DEG | oblique views |
| DeepWeeds | one label per image, rangeland shrubs |
| Northern WA Blue Lupins | single leaves boxed, not plants |
| Wild carrot flowers in canola | flowers from a drone at 6 m |
| CottonWeedDet12 | 29 GB, hand-held phones and cameras at varying heights |
| Precision Sustainable Ag OpenCV 2021 | greenhouse pots; OPPD showed pots transfer badly |
| PAGS8, Palmer amaranth (x2) | Palmer amaranth only, a weed not established in Europe |
| Multiple weed species detection (Cornell, Kansas) | GoPro at varying heights and angles; a candidate for *Chenopodium album* (464 boxes) if the views turn out top-down enough |
| RoboWeedMap | family labels only (Brassicaceae, Poaceae); its licence field is just a company URL |
| Amsinckia, Fireweed, Emerging ryegrass seedlings | species of no use to us, or 14-20 images |
