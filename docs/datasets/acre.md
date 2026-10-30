# ACRE Crop-Weed Dataset

- Source: Zenodo record 8102217 (https://doi.org/10.5281/zenodo.8102217), version 1.0.0,
  June 2023. Datasheet: `ACRE_datasheet.md` in the download.
- Owner: METRICS project (H2020, grant 871252): Politecnico di Milano, INRAE Montoldre, LNE.
  Curator: Riccardo Bertoglio, Politecnico di Milano.
- Licence: **CC BY 4.0** (stated on the Zenodo record, not in the files). Commercial use is
  allowed with attribution: cite the paper "The ACRE Crop-Weed Dataset for Benchmarking Weed
  Detection Models on Maize and Beans Fields" (ECML PKDD 2023 workshop, published 2025 in
  Communications in Computer and Information Science).

## Content

- 1,000 top-down RGB images, 2046×1080, from a teleoperated skid-steer robot with a Basler
  acA2000-50gc camera, rectified. INRAE experimental field, Montoldre, France, 8-10 June 2022.
- 10 recording batches of 100 consecutive images, each on a different plot: 5 in beans, 5
  in maize, under diffuse and direct sunlight (some show the robot's shadow). Neighbouring
  images can show the same plant, so the split is by batch.
- One polygon per plant (`clipping`), a stem point where visible, a class (crop, weed or
  "unknow") and a name, in French. 56,530 plants, about 46 per image.
- Sown species: maize, beans, *Lolium perenne*, *Sinapis arvensis*, *Matricaria chamomilla*,
  *Chenopodium album*. A few natural weeds were named by the annotators (purslane,
  *Digitaria*, *Setaria*). Label map: `configs/label_maps/acre.csv`.

| Label | Species | Plants |
|---|---|---|
| chenopode | *Chenopodium album* | 46,636 |
| haricot | *Phaseolus vulgaris* (crop) | 4,119 |
| mais | *Zea mays* (crop) | 1,821 |
| ray-grass | *Lolium perenne* | 1,002 |
| matricaire | *Matricaria chamomilla* | 954 |
| moutarde | *Sinapis arvensis* | 710 |
| inconnu, empty, class "unknow" | one plant of unknown species (`1PLAK`): ignore regions for the species classes, plants for the two-stage detector | 1,244 (757 at least 20 mm) |
| of which class "weed" | source labels `inconnu (weed)`, `unnamed (weed)`: none of the crops below | 1,083 (635) |
| of which class "crop" | `inconnu (crop)`, `unnamed (crop)`: one of the crops below | 31 (31) |
| others | faba bean, pea, purslane, *Digitaria*, *Setaria* | under 25 each |

## Conversion choices

- `scripts/convert_acre.py` (`configs/acre.yaml`), shared pipeline `weedseg.data.convert`.
- **Group = recording batch.** Only 10 groups, so the 70/15/15 split comes out as 6/2/2
  batches. The download's `split_dictionary.json` is not used. Both validation batches are
  bean plots, so bean is the crop there: `configs/rules/example.yaml` has listed it as a crop
  since 2026-10-23 (see [Decisions](../maskrcnn.md#decisions)).
- **mm/px is estimated at 0.5**: the camera height and lens are not given. Maize 10-15 cm
  tall spans a median 490 px, beans 8-10 cm about 260 px. The real value would come from the
  curator; it matters for the 20 mm size filter.
- Polygons with fewer than 3 points (box used as mask) and plants marked untrusted become
  ignore regions, like unidentified plants.
- **Plants without a species keep the annotators' class.** A plant named "inconnu" or
  without a name still has a class, crop or weed, so its source label carries it (e.g.
  `inconnu (weed)`). In the label map, a weed is `none_of` the dataset's crops (bean, maize,
  faba bean, pea) and a crop `one_of` them, so the classifier learns what that rules out
  ([partial labels](../data-format.md#partial-labels)). The faba beans and peas are classed
  crop by the annotators, in bean plots and (3 faba beans) in a maize plot. A plant of
  class "unknow" stays `unknow`, whatever its name, and rules out nothing.

## Known limitations

- ***Chenopodium album* dominates**: 83% of the plants, in 991 of 1,000 images.
- **Outline quality varies**: 18% of weed polygons have only 3-4 points, mostly tiny
  seedlings (42% of plants under 40 px, 11% of larger ones). Ryegrass blades are often
  traced as separate thin slivers.
- **Little variety within the dataset**: one field, three days.
- Some small seedlings are not annotated; most fall under the size filter anyway.
- Companion dataset: ROSE (same field, 2019-2021, about 3,000 images, similar
  annotations), https://www.challenge-rose.fr/en/home. Not used yet.
- **Related: [VCD](vcd.md)**, partly recorded at INRAE Montoldre too (2019-2020, another
  camera), with the same sown weeds; it boxes the crops only. See
  [provenance.md](provenance.md).
