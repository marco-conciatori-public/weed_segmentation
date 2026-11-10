# Datasets checked and not used

Every dataset we looked at and set aside, with the reason, so none is checked again by
mistake. Before looking into a "new" dataset, search this page for its name, its authors and
its site. Datasets we use are in [provenance.md](provenance.md); the detector's criteria are in
[detector.md](../detector.md#datasets).

The detector needs a **box (or an ignore region) for every plant**, species not needed, from
a camera looking **straight down**: a plant without a box is learned as soil. The images'
**scale** must also be near our datasets' (~0.15-1.3 mm/px, OPPD to CornWeed's BoniRob): the
detector resizes images by pixels, not by mm, so a plant's size in pixels has to match. The segmentation line on `master`
also needed species and outlines, so some sets it dropped would pass here; they are marked as
such below.

## Rejected after downloading them and looking at the labels

The labels were drawn on sample images (`scripts/preview_annotations.py`) and the dataset
deleted afterwards.

| Dataset | Source | Checked | Why not |
|---|---|---|---|
| Tobacco Aerial Dataset (NUST, Mardan, Pakistan; drone at 4 m, ~1 mm/px; CC BY 4.0) | Mendeley 5dpc5gbgpz | 2026-11-08 | Campaigns 1-2 label the weeds per pixel as one continuous carpet, without single plants; campaigns 3-8 label only the tobacco, the weeds being left as background. |
| Sudars (Latvia, Data in Brief 2020; greenhouse pots in Jelgava and three organic vegetable farms; CC BY 4.0) | Mendeley nj4vtk4tt6 | 2026-11-09 | In the field images (935 of 1,176, RealSense 1280x720) only the small weed seedlings are boxed (6,041 weeds, 7 crops): grown radishes, large weeds and some seedlings have no box. In the greenhouse pots some seedlings have no box either. |
| WSD, Weed Stem Detection (Shanghai Artificial Intelligence Laboratory and partners, AAAI 2025, arXiv 2502.06255; Apache 2.0) | ModelScope JinzheAI/WeedStemDetection, `labelled.zip` (14.6 GB) | 2026-11-09 | A weed's box is a small square at its centre (median 45 px, 5-95%: 27-74 px), whatever the weed's size: a grass weed whose leaves spread 200-400 px gets a box around its base, and some plants have no box. Crop boxes are whole plants. Larger ignore regions around the weeds don't fix it: they would need squares of ~700 px around each of ~7.5 weeds per image. |

What else we learned about WSD, in case it comes up again: China, three experimental fields
(maize, soybean, mung bean; the 100 mung bean images are not in the release), weeds sown on
purpose; a vehicle's FLIR BFS-U3-123S6C-C straight down at 1 m, 2048x2048 BMP, about
0.6 mm/px (estimated from the 825 px between the two crop rows of an image and from leaf
sizes; the lens isn't stated); 1,456 labelled images from six days between 30 November and
25 December 2023, 18,512 boxes (10,958 weeds, 5,929 maize, 1,625 soybean), each weed also with
its stem point; consecutive images barely overlap. The 5,605 unlabelled images (32 GB) were
never fetched.

## Rejected after looking at sample images (tier 2, 2026-11-10)

Checked strictly, at Marco's request ("be a bit strict"). A few images per dataset were
fetched (single files, or members read by byte range from the archive), their labels drawn on
them, and the samples deleted afterwards; nothing was converted.

| Dataset | Source | Sample | Why not |
|---|---|---|---|
| NDSU weed-crop robot set (Upadhyay et al., Data in Brief 60 (2025) 111486; North Dakota State University; CC BY 4.0) | Mendeley mthv4ppwyw v2: 1,120 JPGs + YOLO files, 7.8 GB, one folder per crop | 3 images per crop (24) | Straight down, crops boxed, but **the fields' own weeds are not**: only the five weed species grown in greenhouse pots and set between the rows are boxed (one sugar beet image has ~10 unboxed weeds, grasses included); crop plants of the next rows at the image edges and seedlings are unboxed too. It is [ImageWeeds](imageweeds.md)' trial (same three sites, 2021-2022, same five weeds), seen from two remote-controlled robots with a Canon EOS 90D instead of a drone. A "positives only" candidate. |
| Maize-Weed Image Dataset, Nigeria (Olaniyi et al., Federal University of Technology Minna; 18 irrigated farms in Niger State; CC BY 4.0) | Mendeley jjbfcckrsp v1: one 3.8 GB RAR, 500 labelled of 36,874 images | 45 labelled images (the start of the RAR) | **Oblique**: a hand-held Sony A6000 "at a distance of one foot", looking forward and down (the rows recede towards the top of the 1920x1080 frames). **"Weed" boxes cover whole patches** of a weed carpet, not single plants. No article describes it. Useless for a phone detector too, since its boxes aren't plants. |
| SorghumWeedDataset_Segmentation (Justina & Thenmozhi, Data in Brief 52 (2024) 109935; SRM Care Farm, Chengalpattu, Tamil Nadu, India, April-May 2023; CC BY 4.0) | Mendeley y9bmtf4xmr v1: one 3 GB zip, 252 images 6000x4000 (Canon EOS 80D), COCO polygons: sorghum, grasses, broadleaf | 4 test images | Straight down and labelled with care, plant by plant down to cotyledon pairs (only a few tiny plants and edge leaves without a polygon). **Fails on scale**: close-ups "with varying distances" (not given per image), roughly 0.05 mm/px by the cotyledon sizes, each photo ~30 cm of ground, about three times finer than OPPD (0.15), the finest we use; most plants are cotyledon-stage seedlings under the 20 mm limit (median box long side: broadleaf 154 px, sorghum 322 px). Brought to our scale, an image would be ~500 px with a handful of plants. |

## Used only in part

| Dataset | What we use | Why not more |
|---|---|---|
| [Sugar Beets 2016](sugarbeets2016.md) (Bonn, BoniRob) | Part 1 of 11 (one day) as an extra test set, never trained on | Masks per pixel without plant IDs: boxes from the masks' patches merge touching beets and split grasses, and some plants have no mask. PhenoBench's site and group. |
| [GROUNDBASED_WEED](groundbased_weed.md), [VCD](vcd.md), [ImageWeeds](imageweeds.md) | Converted, not in the detector's data | Not every plant boxed: TOMATO_2's tomatoes and plants of uncertain species; VCD's weeds; all but ImageWeeds' potted weeds. They could join with "positives only" training ([detector.md](../detector.md#new-datasets-survey-of-2026-11-06)). |

## Rejected from their papers and pages, not downloaded

From the detector's survey of 2026-11-06:

| Dataset | Why not |
|---|---|
| MH-Weed16 (India, soybean) | Only the weeds are boxed; IEEE DataPort subscription. |
| 3SeasonWeedDet10 (Michigan) | Only the weeds are boxed. |
| CoFly-WeedDB (Greece, cotton, low drone, 207 images) | The cotton is not labelled. |
| USU-Corn-WeedDB (Utah, forage corn, drone, 650 images) | Only three weed species are boxed; the corn is not. |
| CottonWeedDet3, CottonWeedDet12 (Mississippi) | Only the weeds are boxed; hand-held phones and cameras, not straight down, heights vary. |
| LFSD (Beijing, lettuce) | Camera tilted 40-45 degrees. |
| Genze et al., sorghum from a drone | Only the weeds are labelled. |
| Weed-AI's Australian wheat sets | Seedlings under the canopy; the wheat is not boxed. |
| CWD30 | Image-level labels only; oblique views; non-commercial. |
| VegAnn | Two classes per pixel (vegetation, background), no single plants. |
| RiceSEG | Classes per pixel, no single plants. |
| GrassClover | Pasture swards, mostly synthetic images. |
| Seedling-counting sets | Only the crop is labelled. |

From the tier-2 check of 2026-11-10 (strict):

| Dataset | Why not |
|---|---|
| Veridis (Universidad de León, Data in Brief 66 (2026) 112820; two farms near León, Spain; HF unileon-robotics/Veridis, CC BY-SA 4.0) | RealSense D455 at 0.77 m **tilted 45 degrees** down; only beet and maize are boxed, the weeds are not labelled, and heavily occluded crop plants were left unlabelled. Its 10,080 images are 3,252 originals plus augmented copies. |
| WeedsGalore (GFZ Potsdam, maize near Marquardt, Germany, 2023; WACV 2025; CC BY 4.0) | Drone at 5 m, **2.5 mm/px**, 600x600 multispectral tiles: 2.5 times coarser than our coarsest drone data (PhenoBench, 1.0 mm/px). A 20 mm plant spans 8 px of real detail; resizing the tiles up makes it bigger, not sharper. |
| Kaggle "Crop and weed detection" (sesame; CC0) | No site, camera, date or scale stated; 1,300 images resized to 512x512, 2,072 boxes (1.6 per image, close-ups). Nothing to record in provenance.md. |
| CWFID (organic carrot farm, Germany, BoniRob, 2013) | 60 images (492 plants): too few to change a training set of ~10k images. |
| Carrot-Weed (Negotino, North Macedonia, 2017) | 39 images (745 plants): too few. |
| Champ et al. 2020, the WeedElec robot's images (Applications in Plant Sciences 8(7) e11373; Zenodo 3906501, CC BY 4.0) | 104 fully labelled images (83 + 21 test): too few, and **ACRE's site** (INRAE Montoldre) with ACRE's crops (maize, bean) and sown weeds, May 2019. "All visible crop and weed specimens were annotated", so it would be clean. |

Dropped earlier for the segmentation line (`master`, 2026-10), where species and outlines were
needed too: WE3DS (CropAndWeed's farm and species, semantic labels only), the ROSE challenge
data (ACRE's site, access by request form), DeepWeeds (not top-down, one label per image),
WeedMap (drone, cm/px), Plant Seedlings (one plant per image), WeedSense (greenhouse pots,
360-degree phone video), GrowingSoy (GoPro on a quad bike, two coarse weed groups), Image
Safari (smartphones), Weed-AI's oblique RadishWheat, lupin leaves, flowers from a drone,
greenhouse PSA and PAGS8 (Palmer amaranth only). WE3DS and ROSE are also in
[provenance.md](provenance.md#checked-and-not-used-for-their-links).

DeepSeedling, also checked in tier 2, is used without its UGA2015 set (Watkinsville, Georgia,
2015), left out after converting: in its weedy stretches many weeds over 20 mm have no box. See
[deepseedling.md](deepseedling.md#decision).
