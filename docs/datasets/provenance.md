# Dataset provenance

Where, when, by whom and how each dataset we use was recorded, in one place, so that a new
dataset can be checked against the others before it is added: same site, same institution,
same years, same camera rig. Overlaps matter for two reasons:

- **Diversity.** A dataset from a site we already have adds less than its size suggests
  (WE3DS was dropped for this, see below).
- **Honest evaluation.** A gain on one dataset from adding a related one may not carry over
  to other fields.

Details and sources are in each dataset's own doc. "~" marks values we estimated or that
the sources give only roughly. Last updated 2026-11-08.

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
| CornWeed | Intel RealSense D435i on a tractor hoe (1280x720) and on the BoniRob platform (640x480), looking down | 0.5 m | ~0.53 / ~0.81 (estimated) | boxes, maize and weeds (no species) | CC BY 4.0 | candidate (2026-11-08) |
| WeedMaize | Canon PowerShot SX540 HS, straight down (platform not given); tiles of the photos | 1.3 m | ~0.36 (estimated) | boxes, maize and 4 weed species with growth stages | CC BY 4.0 | candidate (2026-11-08) |

## Known links

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
  (GROUNDBASED_WEED); leek (VCD); ragweed, kochia, waterhemp and horseweed (ImageWeeds); many
  of OPPD's species. A model may learn the dataset's look along with them.
- **Plants grown in containers**: OPPD (trays) and ImageWeeds (pots set in the field). On
  `master`, species learned mostly from OPPD transferred badly to CropAndWeed's fields.
- **Weed-AI** is a repository, not a source: its datasets come from many groups (see
  [weedai.md](weedai.md)). Its licence field is what the uploader entered: check the
  original release (LincolnBeet's states none).

## Checked and not used, for their links

| Dataset | Link | Why not used |
|---|---|---|
| WE3DS | BOKU experimental farm Groß-Enzersdorf, 2020-21: CropAndWeed's farm, 16 of its 17 species are CropAndWeed's | adds no diversity; semantic labels only |
| ROSE challenge data | INRAE Montoldre, 2019-2021: ACRE's field | ACRE's site; access by a request form; label format not published |
| Sugar Beets 2016 | University of Bonn's BoniRob, a sugar beet farm near Bonn: PhenoBench's area | PhenoBench's area and crop; a candidate for the detector all the same (crop/weed labels are enough), not added yet |
