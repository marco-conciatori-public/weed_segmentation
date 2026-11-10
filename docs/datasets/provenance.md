# Dataset provenance

Where, when, by whom and how each dataset we use was recorded, in one place, so that a new
dataset can be checked against the others before it is added: same site, same institution,
same years, same camera rig. Overlaps matter for two reasons:

- **Diversity.** A dataset from a site we already have adds less than its size suggests
  (WE3DS was dropped for this, see below).
- **Honest evaluation.** A gain on one dataset from adding a related one may not carry over
  to other fields.

Details and sources are in each dataset's own doc. "~" marks values we estimated or that
the sources give only roughly. Last updated 2026-11-10.

**When adding a dataset:** fill a row in both tables, look for the same place, institution,
period or rig in the other rows, and record any link under [Known links](#known-links).

## Where, who, when

| Dataset | Site(s) | Coordinates | Institution | When | Setting |
|---|---|---|---|---|---|
| [CropAndWeed](cropandweed.md) | Austria: 264 commercial-field sessions and 665 sessions on cultivated plots; sites not published (the plots probably at BOKU's experimental farm, Groß-Enzersdorf) | ~48.20 N, 16.56 E (the farm) | AIT Austrian Institute of Technology | March-July, four seasons (years not given) | fields and plots, 16 crops |
| [ACRE](acre.md) | INRAE Montoldre, France | 46.34 N, 3.43 E | METRICS project: Politecnico di Milano, INRAE, LNE | 8-10 June 2022 | experimental plots: maize, bean; sown weeds |
| [PhenoBench](phenobench.md) | Campus Klein-Altendorf near Bonn, Germany | ~50.62 N, 6.99 E | University of Bonn | 15 May, 26 May, 5 June 2020 (labelled) | one sugar beet field |
| [OPPD](oppd.md) | Denmark (site not checked) | | Aarhus University | 2017-2019, 4 trials | trays, one sown weed species each |
| [GROUNDBASED_WEED](groundbased_weed.md) | Santa Amalia (Badajoz), Spain: five commercial fields | ~39.0 N, 6.0 W | CSIC, Centre for Automation and Robotics | May-June 2021 and 2022 | commercial tomato fields |
| [LincolnBeet](lincolnbeet.md) | near Lincoln, UK: fields "Near 30" and "BBRO" (commercial-like), and a walled garden plot | 53.23 N, 0.53 W (one point for all) | University of Lincoln | 22 April - 4 June 2021, 4-12 days per field | sugar beet fields |
| [ImageWeeds](imageweeds.md) | North Dakota, USA: Agronomy Seed Farm (Casselton), Carrington Research Extension Center, Grand Farm | 46.90 N, 97.21 W; 47.37 N, 99.20 W; 46.72 N, 96.83 W | North Dakota State University | May-Aug 2021, Aug-Sept 2022 | weed trial: seven crops side by side, **potted weeds** set in a line in the centre |
| [VCD](vcd.md) | France: INRAE Montoldre; CTIFL Prigonrieux; Fermes Larrère, Liposthey; Bordeaux Sciences Agro | 46.34 N, 3.43 E; 44.85 N, 0.41 E; 44.43 N, 0.86 W; 44.79 N, 0.61 W | Université de Bordeaux (IMS, Bordeaux Sciences Agro, ENSEIRB-Matmeca), CTIFL | 2019-2021, 9 days | rows of maize, bean, leek; sown and natural weeds |
| [CornWeed](cornweed.md) | one field, "Langsenkamp" (the file names), Germany; the paper names no place | not given | DFKI and Hochschule Osnabrück | 22 July - 2 August 2021 (labelled: 22 and 27 July) | maize rows, natural weeds |
| [WeedMaize](weedmaize.md) | Badajoz, Spain: commercial maize fields | 39.02 N, 6.06 W | Universidad Politécnica de Madrid; CSIC-UPM Centre for Automation and Robotics | April 2020 | commercial maize, V1-V4, natural weeds |
| [DeepSeedling](deepseedling.md) | Corpus Christi, Texas (TAMU2015) and Watkinsville, Georgia (UGA2015, not used; UGA2018), USA; no fields named | not given | University of Georgia (Bio-Sensing and Instrumentation Lab), with Texas A&M | 12 April 2015, 15 June 2015, 13 June 2018 | cotton plots (cultivar and breeding trials), 7-11 days after planting |
| [Sugar Beets 2016](sugarbeets2016.md) | Campus Klein-Altendorf near Bonn, Germany | ~50.62 N, 6.99 E | University of Bonn (and University of Freiburg) | spring 2016, about three days a week (part 1: 23 May) | one sugar beet field |

## How

| Dataset | Platform and camera | Height | mm/px | Labels | Licence | In the detector's data |
|---|---|---|---|---|---|---|
| CropAndWeed | hand-held full-frame SLR, 50 mm, "approximate top-down" | ~1.1 m | ~0.41 | boxes, semantic masks (we derive instances), stem points; 58 weed and 16 crop species | non-commercial, models included | yes |
| ACRE | teleoperated robot, Basler acA2000-50gc, rectified | not given | ~0.5 | polygons, stem points; crops and 4 sown weeds by species | CC BY 4.0 | yes |
| PhenoBench | drone, PhaseOne iXM-100, 80 mm | ~21 m | 1.0 | instance masks; beet, and weeds without species | CC BY-NC-SA 4.0 | yes |
| OPPD | fixed rig over trays | not checked | 0.15 | boxes, 46 species, daily time series | CC BY-NC-SA 4.0 | yes |
| GROUNDBASED_WEED | hand-held Canon PowerShot SX540 HS | 1-1.5 m | ~0.3 | boxes, tomato and 5 weeds; TOMATO_2's tomatoes unlabelled | ODbL 1.0 | no: not every plant boxed |
| LincolnBeet | frames of 4K videos, two cameras (12 MP 26 mm, 64 MP 29 mm, equivalent focal lengths) | ~0.5 m | ~0.35 | boxes, beet and weeds without species | none in the original; CC BY 4.0 per Weed-AI | yes |
| ImageWeeds | drone, DJI Phantom 4 Pro V2, looking down (Weed-AI says 70 degrees; the paper gives no angle) | ~3.7 m | ~1.0 | boxes, 5 weed species, potted only; crops and field weeds not boxed | CC BY 4.0 | no: not every plant boxed |
| VCD | covered, lit module; Basler acA2500, 4 mm lens (current module) | 35-45 cm | ~0.23-0.9 (images resized) | boxes of crops and their stems; weeds not boxed | CC BY 4.0 | no: not every plant boxed |
| CornWeed | Intel RealSense D435i on a tractor hoe (1280x720) and on the BoniRob platform (640x480), looking down | 0.5 m per the paper; the images suggest ~1 m (tractor) | ~1.07 / ~1.3 (estimated from the row distance, see cornweed.md) | boxes, maize and weeds (no species) | CC BY 4.0 | yes, from `public6` (2026-11-09) |
| WeedMaize | Canon PowerShot SX540 HS, straight down (platform not given); tiles of the photos | 1.3 m | ~0.36 (estimated) | boxes, maize and 4 weed species with growth stages | CC BY 4.0 | test only: held out whole as the unseen-site test (2026-11-09) |
| DeepSeedling | video frames: Samsung Galaxy Note3 phone by hand (TAMU2015), Panasonic DMC-G6 and Fujifilm X-A10 on a cart (UGA2015, UGA2018); close to straight down | ~0.5 m | ~0.4 / ~0.35 / ~0.35 (estimated, see deepseedling.md) | boxes, cotton and weeds (no species); seedlings under ~30 px on the labelling screen unboxed, and in UGA2015 many weeds | GPL 3.0+ | TAMU2015 and UGA2018, from `public7` (2026-11-10); UGA2015 not used |
| Sugar Beets 2016 | BoniRob robot, JAI camera (1296x966, 8 mm lens), straight down | ~85 cm | ~0.33 (the paper's; its field of view gives 0.24) | masks per pixel, beet and weed, no plant IDs | CC BY-SA 4.0 | no: part 1 only, as an extra test set |

## Known links

- **PhenoBench and Sugar Beets 2016: same site and group.** Both are sugar beet fields at
  Campus Klein-Altendorf, by the University of Bonn: Sugar Beets 2016 from the BoniRob
  robot in 2016, PhenoBench from a drone in 2020. Sugar Beets 2016 is only a test set.

- **GROUNDBASED_WEED and WeedMaize: same group, camera and region.** D. Andújar's group
  (CSIC-UPM) photographed both in Badajoz province with a Canon PowerShot SX540 HS: tomato
  fields in 2021-22, maize fields in 2020. Their weeds overlap (*Cyperus rotundus*,
  *Solanum nigrum*, *Echinochloa crus-galli*, *Portulaca oleracea*).

- **ACRE and VCD: same site, same sown weeds.** Both were recorded at INRAE Montoldre, and
  VCD's sown weeds (mustard, ryegrass, chamomile, fat-hen) are ACRE's. VCD's Montoldre images
  are from 2019-2020, ACRE's from June 2022, with other cameras. A gain on ACRE from VCD
  may be partly site-specific.
- **Sugar beet from three places**: CropAndWeed (Austria), PhenoBench (Bonn, drone) and
  LincolnBeet (Lincoln, close-up video). PhenoBench and LincolnBeet both label weeds without
  species.
- **Maize and bean from several places**: CropAndWeed, ACRE and VCD (both crops), ImageWeeds
  (unboxed maize and soybean plots).
- **Species from one source only**: tomato, *Cyperus rotundus* and *Setaria verticillata*
  (GROUNDBASED_WEED); leek (VCD); ragweed, kochia, waterhemp and horseweed (ImageWeeds); cotton
  (DeepSeedling, the only set from the US South and the only phone video); many of OPPD's
  species. A model may learn the dataset's look along with them.
- **Plants grown in containers**: OPPD (trays) and ImageWeeds (pots set in the field). On
  `master`, species learned mostly from OPPD transferred badly to CropAndWeed's fields.
- **Weed-AI** is a repository, not a source: its datasets come from many groups (see
  [weedai.md](weedai.md)). Its licence field is what the uploader entered: check the
  original release (LincolnBeet's states none).

## Checked and not used, for their links

Every dataset set aside, with the reason, is in [rejected.md](rejected.md); these two are
here for their links to datasets we use.

| Dataset | Link | Why not used |
|---|---|---|
| WE3DS | BOKU experimental farm Groß-Enzersdorf, 2020-21: CropAndWeed's farm, 16 of its 17 species are CropAndWeed's | adds no diversity; semantic labels only |
| ROSE challenge data | INRAE Montoldre, 2019-2021: ACRE's field | ACRE's site; access by a request form; label format not published |
