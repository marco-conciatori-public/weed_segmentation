# Internal data format

Every dataset, public or our own, is converted to the same COCO-based format. Training and
evaluation code then reads only this format. Per-dataset notes live in [datasets/](datasets/).

Status: proposed and in use by the CropAndWeed converter. It has not yet been agreed with
the team that prepares our own dataset.

## Why COCO with instance masks

- The label format (boxes or instance masks) is not decided yet. Boxes can be computed from
  masks, but masks can't be recovered from boxes, so masks keep both options open.
- Masks also help to locate the plant's growing point for the weeding tool.
- Most annotation tools (CVAT, Label Studio) and training frameworks read and write COCO.

## Layout

```
data/raw/<dataset>/              original download, untouched
data/processed/<dataset>/<taxonomy>/{train,val,test}.json
configs/<dataset>.yaml           download and conversion settings
configs/label_maps/<dataset>.csv source label -> categories
```

Images are not copied. `info.image_dir` in each JSON gives their folder, relative to the
project root.

## COCO fields

Standard COCO fields, plus these extras:

| Where | Field | Meaning |
|---|---|---|
| image | `group` | Split unit, e.g. a recording session. All images of a group are in the same split. |
| image | `session`, `subset`, ... | Dataset-specific metadata, e.g. soil, moisture, lighting. |
| image | `mm_per_px`, `mm_per_px_estimated` | Ground resolution, and whether it is only an estimate. |
| annotation | `segmentation` | Instance mask as compressed RLE. |
| annotation | `stem` | `[x, y]` of the plant's stem / growing point, when known. |
| annotation | `source_label` | The label as written in the source dataset. |
| annotation | `iscrowd` | 1 = **ignore region**, see below. |
| annotation | `hidden_plant` | Set by combining on plants that are ignore regions only because of the choice of classes (rule 3 below); the two-stage detector learns them as plants. |
| category | `supercategory` | `crop`, `weed`, `mixed` or `ignore`. |
| category | `source_labels`, `common_names`, `scientific_names` | What the category was built from. |

## Categories and taxonomies

- Each dataset has a label-map CSV with one row per source label. It has the columns
  `source_label`, `role` (crop/weed), `eppo`, `eppo_note`, `scientific_name`, ... and
  optionally one column per dataset-specific **taxonomy** (a class grouping). The
  conversion config picks the taxonomy.
- The `species` taxonomy uses **EPPO codes** as category names (e.g. `CHEAL` =
  *Chenopodium album*). That gives every dataset the same key for the same species.
  Genus-level labels use EPPO genus codes (e.g. `1GERG` = *Geranium*).
- The taxonomies **`genus`, `family` and `order`** work for every dataset without extra
  columns. They group weeds by their EPPO taxonomy ([reference-data.md](reference-data.md)),
  which handles rare species and datasets labelled at different levels (*Poa* vs *Poa
  annua*).
  - **Crops are never grouped.** They keep their own code, since a crop must never share
    a class with a weed (by family, maize would join the grass weeds and sugar beet the
    goosefoots).
  - A weed whose group is also a crop's code keeps its own code too (e.g. crop poppy
    labelled *Papaver* vs the weed *Papaver rhoeas*).
- Dataset-specific groupings (e.g. CropAndWeed's `fine24`) are label-map columns.
  Choosing a coarser grouping only takes a re-conversion, never relabelling.
- **Plants of unknown species** use the code of the plant kingdom, `1PLAK` (*Plantae*), e.g.
  PhenoBench's weeds, which are labelled one plant at a time but only as "weed". They
  can't train the species classes (they become ignore regions there, rule 3 below), but
  the plant detector of the two-stage pipeline learns from them ([twostage.md](twostage.md)).
  Use `1PLAK` only when each label is one whole plant. Its role may be `unknown` (e.g.
  ACRE's "unknown" plants, some of which the annotators called crops). Unclear areas
  (e.g. CropAndWeed's "Vegetation") get no code and stay ignore regions everywhere.

## Combining datasets and choosing the classes

Each dataset is converted on its own with taxonomy `species` (one category per EPPO code),
the same `min_plant_size_mm`, and its own split. `scripts/combine_datasets.py`
(`configs/combine.yaml`) then merges them and chooses the classes on the combined training
images, so that every dataset counts (`weedseg.data.combine`):

1. A species with at least `min_train_images` training images is a class.
2. A rarer species goes to `other_plant`.
3. A label coarser than species (e.g. genus only) goes to `other_plant`, unless one of its
   species is a class: then it becomes an ignore region, since the plant may well be of
   that species (e.g. *Geranium* next to *Geranium pusillum*). A code above order (clade,
   class, phylum, kingdom, e.g. `1PLAK`) always counts as holding a class, since the taxa
   table doesn't record its species.
4. Ignore regions stay ignore regions.

The decisions are printed and saved to `classes.txt`. The output goes to
`data/processed/<datasets joined by +>/species_min<N>img_min<size>mm`. Each plant keeps its
own EPPO code in an `eppo` field, whatever class it ends up in; image file names become
paths from the project root (`info.image_dir` is "."), and `info.licences` lists the
licences of all the datasets, since the most restrictive one applies to the combination.

## The other_plant category

With per-species classes, species too rare to learn and labels coarser than species (e.g.
genus only) share one `other_plant` category (supercategory `other`). It is a normal class:
the model learns to find these plants, and at inference they go to a generic model. It is
not for unidentified plants: those may belong to any class and stay ignore regions.

## Ignore regions

An annotation has `iscrowd = 1` when it should neither be learned as an object nor be
treated as background:

- plants the source dataset could not identify (e.g. CropAndWeed "Vegetation"),
- labels left unmapped by the chosen taxonomy,
- annotated plants whose mask could not be derived,
- vegetation present in a mask but without any box.

An ignore region keeps the category of its label when that label has one, and uses the
`ignore` category otherwise. Training code must drop ignore regions from the targets and should not
penalise predictions on them. **Evaluation:** COCO's standard evaluation ignores crowd
regions only for predictions of the same category, so ignore regions have to be matched
against predictions of any class, whatever category the region kept (`weedseg.evaluation`).

## Splits

Splits are made by `group` (recording session, field, date), never per image. Neighbouring
frames of one session are near-duplicates and would leak into the test set. The split
fractions and the seed are in the dataset config.
