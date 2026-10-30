# Internal data format

Every dataset, public or our own, is converted to the same COCO-based format. Training and
evaluation code then reads only this format. Per-dataset notes live in [datasets/](datasets/);
[datasets/provenance.md](datasets/provenance.md) compares where and how each was recorded, to
check a new dataset against the others before adding it.

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

## Downloading

`scripts/download_datasets.py` fetches the datasets listed in `configs/download.yaml`; each
dataset config's `download` section lists its files (URL, size, checksum when published,
whether to unpack, and `extract_to`, a subfolder to unpack into, for archives with the same
layout such as Weed-AI's), or folders of a Nextcloud public share (`share` link, folder
`path`, total size). It is built for the slow, unstable connection we have in China (VPN):

- Each file comes in pieces over several connections at once (`transfer.connections`). On
  2026-10-21 one connection to phenobench.org gave ~0.66 MB/s, six gave ~3 MB/s.
- Progress is saved every few seconds (`<file>.part.json` next to `<file>.part`). Stop the
  script, lose the network or crash: running it again continues where it stopped.
- A connection that slows to a trickle is reopened, and failures are retried with growing
  waits, by default without limit (`max_retries: null`).
- A file that brings no new byte for `give_up_after_min` (30) minutes, because the server is
  down or can't serve part of it, is set aside with its progress kept, and the next file
  starts. Set-aside files get another pass after the others, as long as the previous pass
  brought anything new. Errors a retry can't fix (wrong URL, a web page instead of the file)
  stop that file at once, not the others. The unfinished files are listed at the end.
- Sizes and MD5/SHA-256 are checked before a file is renamed into place, and unpacking a zip
  checks every member's CRC. A file that fails is kept as `.bad` and fetched anew next time.
- Finished files are skipped (markers in `<raw_dir>/.downloaded/`). An unfinished download
  from elsewhere (e.g. a browser) can be renamed to `<raw_dir>/.archives/<name>.part` and is
  continued.
- A share folder is listed through the share's WebDAV address and copied into
  `<raw_dir>/<path>` file by file, several files at once (one connection each); a file
  continues from its `<file>.part` after a break. Files already there with the listed size
  are skipped. The share's own "download" button would build a zip on the fly, which can't be
  continued after a break.

Server quirks found on 2026-10-21: digital.csic.es (GROUNDBASED_WEED) sits behind a bot check
that challenges browsers but lets scripts through, and takes ~30 s before each answer starts,
hence the long timeout and large pieces. gitlab.au.dk (OPPD) was down for maintenance.
hf-mirror.com only redirects to huggingface.co, so it brings nothing. OPPD comes from
Dataset Ninja's copy. Its Dropbox link was disabled for too much traffic on 2026-10-22 (the
script then reports a web page instead of the file); the link behind the download button on
datasetninja.com worked, at ~2 MB/s (about 6 hours for 45 GB).

GROUNDBASED_WEED's `TOMATO_2.zip` stopped at 512 MiB (25% of 2.1 GB) on 2026-10-21 for a
browser and this script alike, on two computers. The script's pieces starting at 512 MiB or
later got no byte at all, so the server seems unable to serve the rest of the file. Not
confirmed yet: on 2026-10-22 digital.csic.es stopped answering altogether, from China and
from elsewhere. `TOMATO_1.zip` stopped at 512 MiB too. On 2026-10-26 the authors sent a
Nextcloud share (saco.csic.es) with both folders unpacked, so these now come from there, file
by file. Its WebDAV listing answers only requests marked as coming from its web page
(`X-Requested-With`); the files themselves have no such check. Each request took ~1.3 s to
start, and one connection gave ~1.7 MB/s.

## COCO fields

Standard COCO fields, plus these extras:

| Where | Field | Meaning |
|---|---|---|
| image | `group` | Split unit, e.g. a recording session. All images of a group are in the same split. |
| image | `session`, `subset`, ... | Dataset-specific metadata, e.g. soil, moisture, lighting. |
| image | `mm_per_px`, `mm_per_px_estimated` | Ground resolution, and whether it is only an estimate. |
| image | `box_only` | The image's dataset is labelled with boxes only, without masks (see below). |
| annotation | `segmentation` | Instance mask as compressed RLE. |
| annotation | `stem` | `[x, y]` of the plant's stem / growing point, when known. |
| annotation | `source_label` | The label as written in the source dataset. |
| annotation | `iscrowd` | 1 = **ignore region**, see below. |
| annotation | `hidden_plant` | Set by combining on plants that are ignore regions only because of the choice of classes (rule 3 below); the two-stage detector learns them as plants. |
| annotation | `one_of`, `none_of` | From the label map: EPPO codes of taxa the plant belongs to one of, or to none of (see below). |
| annotation | `allowed_classes` | Set by combining on hidden plants whose label rules out some classes: the classes they may belong to, a [partial label](#partial-labels) for the species classifier. |
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
  have no class of their own (they become ignore regions in the species files, rule 3
  below), but the plant detector of the two-stage pipeline learns from them
  ([twostage.md](twostage.md)), and so does the species classifier when their label rules
  out some classes ([partial labels](#partial-labels)). Use `1PLAK` only when each label is
  one whole plant. Its role may be `unknown` (e.g. ACRE's "unknown" plants, some of which
  the annotators called crops). Unclear areas (e.g. CropAndWeed's "Vegetation") get no code
  and stay ignore regions everywhere.
- **What a coarse label still says** goes into the optional columns `one_of` and `none_of`:
  EPPO codes separated by spaces, of rank species to order (the taxa table records which
  species those hold). The plant belongs to one of the `one_of` taxa and to none of the
  `none_of` taxa. PhenoBench's weeds are `1PLAK` with `none_of` `BEAVA`: whatever they are,
  not sugar beet. OPPD's monocots of unknown species are `one_of` the 11 monocot orders, its
  dicots `none_of` them (EPPO has no single monocot taxon: its monocots are two clades,
  Commelinids and Basal monocotyledons). ACRE's plants without a name keep the annotators'
  class in their source label, e.g. `inconnu (weed)`, which is `none_of` the dataset's
  crops. A genus or family label needs neither column: the taxonomy says which classes it
  holds. The converter copies the columns into each annotation of that label.

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

The decisions are printed and saved to `classes.txt`, under the list of datasets. The output
goes to `data/processed/<name>/species_min<N>img_min<size>mm`, where `<name>` is the
config's `name` (since 2026-10-30 `public7`) or else the dataset names joined by + (the older
combinations, e.g. `cropandweed+acre`): a short name keeps run and preview paths under
Windows' 260 characters. Each of the config's `variants` (the datasets plus some more) is
combined the same way into a folder of its own, for classifier variants trained on other
data (`public7+imageweeds`). Each plant keeps its
own EPPO code in an `eppo` field, whatever class it ends up in; image file names become
paths from the project root (`info.image_dir` is "."), and `info.licences` lists the
licences of all the datasets, since the most restrictive one applies to the combination.

### Partial labels

A plant hidden by rule 3 has no class, but its label may still rule out some classes.
Combining then gives it `allowed_classes`, the classes it may belong to, and the species
classifier learns from it that it is one of those, without being told which
([twostage.md](twostage.md#partial-labels); added 2026-10-26 after the classifier had
learnt to call every PhenoBench-like plant sugar beet):

- A genus, family or order code allows the classes in that taxon: CropAndWeed's *Poa* can
  be `POAAN` or `other_plant`. When the label is a crop or a weed, only classes of the same
  role: CropAndWeed's crop poppy (*Papaver*, genus only) is not the weed *Papaver rhoeas*,
  so it can only be `other_plant`.
- A code above order allows every class, narrowed down by `one_of` and `none_of`.
- `other_plant` is always allowed (a species without a class of its own), unless `one_of`
  lists only species that are all classes.
- A plant whose label allows every class (ACRE's "unknow", GROUNDBASED_WEED's `NR`) gets no
  partial label: it stays a plain ignore region for the classifier.

The partial labels of the training split are listed at the end of `classes.txt`, and
`info.partial_labels` counts them in each split. Mask R-CNN and every evaluation of
detections still treat these plants as ignore regions.

## Datasets labelled with boxes only

Some datasets name each plant's species but give only its box (GROUNDBASED_WEED, OPPD, and
the Weed-AI datasets: LincolnBeet, ImageWeeds, VCD). Their converter passes `box_only=True`
to `convert_dataset`, which marks every image `box_only`.
They are combined with the others into the same files, with one class list (decided
2026-10-21), and each step takes what it can use:

- Their plants count when combining chooses the classes: a species may become a class only
  thanks to them.
- The species classifier of the two-stage pipeline learns from their plants, since a crop
  needs only a box, and `evaluate_classifier.py` scores it on them
  (`accuracy_by_dataset.csv`).
- Mask R-CNN (the species model, the two-stage detector) leaves these images out, since it
  would learn the boxes as plant outlines. So does the evaluation of instance predictions
  (`evaluate_maskrcnn.py`, `predict_twostage.py`, `review_errors.py`): all of them read the
  files through `weedseg.maskrcnn.data.load_masked_coco`.

A labelled plant without a mask on an image not marked `box_only` stops Mask R-CNN training
and evaluation with an error, so a converter that forgets the mark is noticed.

## The other_plant category

With per-species classes, species too rare to learn and labels coarser than species (e.g.
genus only) share one `other_plant` category (supercategory `other`). It is a normal class:
the model learns to find these plants, and at inference they go to a generic model. It is
not for unidentified plants: those may belong to any class and stay ignore regions (with a
partial label for the classifier when their label rules out some classes).

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
fractions and the seed are in the dataset config. A dataset's own split is kept instead when
it is cleaner than ours, e.g. PhenoBench's, which separates crop rows
(`convert_dataset(splits=...)`).
