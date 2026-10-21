# CropAndWeed

Chosen as the first dataset for the prototype, while our own dataset is being prepared.
Facts below were checked against the paper, its supplement, the licence and the downloaded
annotations on 2026-10-13, unless marked otherwise.

- Paper: Steininger et al., *The CropAndWeed Dataset: a Multi-Modal Learning Approach for
  Efficient Crop and Weed Manipulation*, WACV 2023 (paper and supplementary material on the
  CVF open-access site).
- Code and download: <https://github.com/cropandweed/cropandweed-dataset>. The data itself is
  hosted on vitro-testing.com and fetched by `scripts/download_datasets.py`.
- Owner: AIT Austrian Institute of Technology.

## What we need from a dataset, and how CropAndWeed fits

| Project requirement | CropAndWeed | Fit |
|---|---|---|
| Top-down (nadir) view | Hand-held SLR, "approximate top-down perspective" | Good, but not a fixed rig |
| ~1 mm/px or finer | ~0.41 mm/px (estimate, see below) | Finer than needed; must be resampled to match our data |
| Weeds labelled by **species** | 58 weed + 16 crop species, some only to genus | Good, the best public match we know |
| Boxes or instance masks | Boxes + **semantic** masks + stem points | Instance masks derived by us (see below) |
| Growing point for the tool | Stem point per plant | Good |
| Field conditions, many sites | 264 commercial-field sessions plus 665 cultivated-plot sessions, Austria, 4 seasons | Good, but one country only |
| Licence usable for the project | Non-commercial only | OK for research; see the licence section below |

## Licence

From the `LICENCE` file in the GitHub repository:

- **Allowed:** non-commercial use (academic research, teaching, scientific publications,
  personal experimentation), by academic and non-academic entities. Publishing derived works
  that are "abstract representations" of the data, such as trained models, as long as the
  dataset can't be recovered from them.
- **Not allowed:** redistributing the dataset or modified versions of it. Using the dataset
  **or any derivative work (trained models included) for commercial purposes**.
- **Required:** citing the dataset in any work that uses it. The licence text says "Apron
  Dataset", apparently copied from another AIT dataset; cite the CropAndWeed WACV 2023 paper.

What this means for NeutraWeed: fine for prototyping, experiments and publications. A model
trained on CropAndWeed must not end up in a partner's commercial product. Keep a record of
which datasets each model was trained on, so it can be retrained without CropAndWeed if
needed. Don't put CropAndWeed files in shared data drops to partners; share the download
script instead.

## Content

| | Application Set (`ave-`) | Experimental Set (`vwg-`) | Total |
|---|---|---|---|
| What | Commercial fields in Austria | Plots sown with chosen species | |
| Sessions | 264 | 665 | 929 |
| Images | 3,044 (234 with no plants) | 4,990 | 8,034 |
| Boxes | 45,076 | 66,877 | 111,953 |

- Of the boxes, 33,000 are the fallback class **Vegetation** (label 255): plants smaller than
  16x16 px or not identifiable. The paper excludes them from training and ignores detections
  on them during evaluation; we do the same (ignore regions).
- Maize and sugar beet are labelled by growth stage (two-leaf ... max). We merge the stages
  into one species and keep the stage as the annotation attribute `growth_stage`.
- A session is one site or plot at one time, ~20 images taken at least 3 m apart.
- Per-instance counts by species are in Tables 6-7 of the supplement. The box counts in the
  download match them exactly.

### Acquisition and resolution

- Semi-professional full-frame SLR, 50 mm lens, auto exposure, ~1.1 m above ground, hand-held
  "approximate top-down", March to July over four years.
- Images are 1920x1088 px. The camera's native resolution is much higher, so these are
  downscaled.
- **mm/px is not given.** Estimate: 36 mm sensor width x 1100 mm / 50 mm = ~792 mm across
  1920 px, so **~0.41 mm/px**. The height is approximate and varies image to image, so the
  real value probably varies by ±20% or more. It is stored as `mm_per_px` with
  `mm_per_px_estimated: true`.

## Raw format

```
images/<name>.jpg                  1920x1088 RGB
bboxes/<name>.csv                  no header; left,top,right,bottom,label_id,stem_x,stem_y
                                   (pixel coordinates, right/bottom inclusive)
labelIds/CropAndWeed/<name>.png    uint8 semantic mask, same label ids, 0 = soil, 255 = Vegetation
params/<name>.csv                  header + one row: moisture (0-2), soil (0-2),
                                   lighting (0-1), separability (0-2)
<name> = <subset>-<session>-<image>, e.g. vwg-0328-0004
```

Label ids and names come from `cnw/utilities/datasets.py` in the GitHub repository; scientific
names from the supplement. Our copy, with EPPO codes and class groupings, is
`configs/label_maps/cropandweed.csv`.

- `name_source = paper_supplement`: scientific name taken from the supplement.
- `name_source = inferred`: the label is defined in the code but has no annotations and no
  scientific name in the paper; we inferred the name from the common name. All these labels
  have **zero instances** in the data, so mistakes there don't matter now.
- EPPO codes are checked automatically against the EPPO taxa table
  ([../reference-data.md](../reference-data.md)). All 88 codes are active. Where the
  paper's name differs from EPPO's, it is either an EPPO synonym (e.g. *Cyanus segetum*
  for `CENCY`, accepted name *Centaurea cyanus*), or explained in `eppo_note`:
  - sugar beet: `BEAVA` is the sugar-beet variety; the paper names the subspecies,
  - "*Polygonum aviculare* agg.": the aggregate; we use the species code,
  - "*Taraxacum* sect. Ruderalia": a section, which has no EPPO code; we use the genus.

The paper mentions time stamps, GPS coordinates and camera parameters as meta-annotations,
but **the public download contains only the four `params` values**.

## How we convert it

`scripts/convert_cropandweed.py` with `configs/cropandweed.yaml`. Output:
`data/processed/cropandweed/<taxonomy>/{train,val,test}.json` (format:
[../data-format.md](../data-format.md)).

- **Instance masks.** The dataset has only semantic masks. A box's mask is the pixels
  inside it that carry its label. Where boxes of the same label overlap (common: 872
  overlapping pairs in a sample of 600 images), each shared pixel goes to the box with the
  nearest stem point. Where leaves of neighbouring plants overlap, the cut is therefore an
  approximation (a straight line between stems), not a true leaf boundary. Checked visually
  on sample images.
- **Boxes without mask pixels** (~0.2% of boxes): kept as ignore regions with a box-shaped
  mask (`mask_from_box: true`).
- **Mask regions without a box**: some masks contain vegetation that has no box (e.g. a
  large cowslip patch). Connected regions of at least 256 px (16x16, the dataset's own
  minimum annotated size) become ignore regions (`unboxed: true`), so a model doesn't learn
  them as background. Smaller leftover fragments are dropped.
- **Size filter:** plants whose box is shorter than `min_plant_size_mm` (20 mm) on its
  longer side become ignore regions, before anything is counted. Images left without
  plants stay as negative samples (`keep_negative_images`).
- **Splits** are made by session, 70/15/15, seed 0, balanced over species: sessions with the
  rarest species are placed first, each in the split least filled with that species and
  with images, relative to the split's share (iterative stratification). A species seen in only one or two sessions still
  can't reach all three splits. The paper also used 70/15/15 but split
  randomly per image and did not publish the split, so **our scores can't be compared
  directly with the paper's**.
- **Taxonomies** (column of the label map, chosen in the config):
  - `species`: EPPO codes, 73 species/genera with instances (the paper's 74 classes, but
    flat-leaf and curly-leaf parsley share one EPPO code). Very long-tailed; see the
    statistics below.
    This is the project's setup: convert with `species` (folder `species_min20mm`), then
    choose the classes with `scripts/combine_datasets.py` (see
    [../data-format.md](../data-format.md)). With CropAndWeed alone, N = 50 and 20 mm: 29
    species classes, each present in train, val and test; *Poa*, *Mercurialis* and
    pumpkins (*Cucurbita*) go to `other_plant`.
  - `fine24`: the authors' 8 crop + 16 weed groups. Some species are left out (e.g.
    courgette, cowslip, purslane, cleavers, wild pansy) and become ignore regions.
  - `crops_or_weed9`, `crop_or_weed2`: coarser variants from the paper.
  - `genus`, `family`, `order`: weeds grouped by EPPO taxonomy, crops kept as species
    (see [../data-format.md](../data-format.md)).

  | Taxonomy | Categories with data | of which under 100 instances |
  |---|---|---|
  | species | 73 | 20 |
  | genus | 63 | 17 |
  | family | 32 | 8 |
  | order | 26 | 7 |

  Counts are over the whole dataset (not per split), Vegetation excluded. Grouping by
  family more than halves the classes and removes most of the rare ones. The remaining
  rare classes are mostly crops, which are never grouped.

Quirks inherited from the upstream mappings (kept as they are, for comparability):
- Crop **poppy** (*Papaver*, label 22) is put in the `fine24` group `poppy` together with the
  weed *Papaver rhoeas*, and in the `weed` class of the two coarse variants.
- "Hedge mustard" appears twice: label 54 (*Sisymbrium*, genus) and label 80
  (*Sisymbrium officinale*).

## Open questions

1. **Which crops and weed species matter to NeutraWeed?** This decides the taxonomy for the
   prototype: species, genus or family (EPPO-based), `fine24`, or a custom grouping made by
   adding a column to the label map. Are the crops in CropAndWeed (maize, sugar beet, soy, sunflower, beans,
   pea, potato, pumpkin) relevant to our fields?
2. **Target resolution.** What mm/px will the robot's camera deliver? CropAndWeed is ~0.41
   mm/px. Bringing it to ~1 mm/px shrinks the image ~2.4x, and the smallest annotated
   plants (16 px) would drop to ~7 px. Should we train at native resolution until our own
   data arrives?
3. **Experimental Set bias.** 62% of the images come from plots sown with one or a few chosen
   species. A model could learn plot context instead of plant appearance. Proposal: report
   test scores on the Application Set (`subset = application`) separately.
4. **Commercial use.** Does the NeutraWeed consortium agreement allow partners to use
   models trained on non-commercial data? This doesn't block the prototype.
5. **mm/px variation.** The camera height is approximate. Is a ±20% scale variation a
   problem, or does scale augmentation cover it? (Probably fine for the prototype.)
6. **Missing metadata.** GPS and time stamps would allow a split by location and date.
   Worth asking AIT whether they can share them? Splitting by session is probably good
   enough for now.

## Statistics after conversion

Full conversion of the annotations (taxonomy `species`, seed 0), run without images on
2026-10-13:

| Split | Images | of which Application Set | Sessions | Instances | Ignore regions | Species present |
|---|---|---|---|---|---|---|
| train | 5,627 | 1,955 | 670 | 53,544 | 22,032 | 69 |
| val | 1,205 | 518 | 125 | 13,261 | 6,613 | 55 |
| test | 1,202 | 571 | 134 | 12,143 | 5,037 | 56 |

- Most frequent in train: *Setaria viridis* (SETVI) 7,078, maize 4,295, sugar beet 3,859,
  soy 3,201, *Poa* 2,902, *Geranium pusillum* 2,500, *Amaranthus retroflexus* 2,033,
  *Chenopodium album* 2,011, *Sonchus arvensis* 1,969, *Cirsium arvense* 1,892.
- Of the 73 species, 12 have fewer than 20 training instances, 4 don't appear in train at all
  and 17 don't appear in test. Rare species are only found in a few sessions, so a
  session-based split can't spread them over all three splits. For species-level evaluation,
  either merge them (e.g. `fine24`) or report only species with enough test instances.
- Conversion of all 8,034 images took ~16 minutes on the dev laptop. The JSON files total
  ~68 MB.
