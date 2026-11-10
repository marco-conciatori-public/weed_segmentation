# Internal data format

Every dataset, public or our own, is converted to the same COCO-based format. Training and
evaluation code then reads only this format. Per-dataset notes live in [datasets/](datasets/);
[datasets/provenance.md](datasets/provenance.md) compares where and how each was recorded, to
check a new dataset against the others before adding it.

This branch (`bounding_box_only`) trains a plant detector with boxes only, so the converted
files hold **boxes, no masks**. A dataset that has outlines (CropAndWeed's semantic masks,
PhenoBench's instance masks, ACRE's polygons) uses them only inside its reader: to find each
plant's box and the ignore regions. The segmentation line (branch `master`) keeps the masks.

Most annotation tools (CVAT, Label Studio) and training frameworks read and write COCO.

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
| image | `dataset` | Set by combining: the source dataset. |
| annotation | `stem` | `[x, y]` of the plant's stem / growing point, when known. |
| annotation | `source_label` | The label as written in the source dataset. |
| annotation | `iscrowd` | 1 = **ignore region**, see below. |
| annotation | `eppo` | Set by combining: the EPPO code of the plant's own label, so the evaluation can say which plants the detector misses. |
| category | `supercategory` | `crop`, `weed`, `mixed`, `plant` or `ignore`. |
| category | `source_labels`, `common_names`, `scientific_names` | What the category was built from. |

`info.taxonomy` names the taxonomy a file was converted with (`species` for the converted
datasets, `plant` for the combined data).

## Categories and taxonomies

- Each dataset has a label-map CSV with one row per source label. It has the columns
  `source_label`, `role` (crop/weed), `eppo`, `eppo_note`, `scientific_name`, ... and
  optionally one column per dataset-specific **taxonomy** (a class grouping). The
  conversion config picks the taxonomy; combining into the detector's data needs `species`,
  so each plant keeps its EPPO code. The species and groupings matter only for the reports
  (plants found per species): the detector itself has one class.
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
  PhenoBench's and LincolnBeet's weeds, which are labelled one plant at a time but only as
  "weed". For the detector they are plants like any other. Use `1PLAK` only when each label
  is one whole plant. Its role may be `unknown` (e.g. ACRE's "unknown" plants, some of which
  the annotators called crops). Unclear areas (e.g. CropAndWeed's "Vegetation") get no code
  and stay ignore regions.

## Combining datasets into the detector's data

Each dataset is converted on its own with taxonomy `species` (one category per EPPO code),
the same `min_plant_size_mm`, and its own split. `scripts/combine_datasets.py`
(`configs/combine.yaml`) then merges them into one class (`weedseg.data.combine`):

- Every labelled plant becomes `plant`, whatever its label: a species, a genus only, or
  `1PLAK`. It keeps its own code in `eppo`.
- Ignore regions stay ignore regions (category `ignore`).
- Each dataset keeps its own split; image file names become paths from the project root
  (`info.image_dir` is "."), each image gets its `dataset`, and `info.licences` lists the
  licences of all the datasets, since the most restrictive one applies to the combination.
- Test-only datasets (`test_only`) are never trained on: all their images, from whichever of
  train/val/test.json they have, go to `test_<dataset>.json` next to the combined splits, to
  be scored on their own (`evaluate.split` of `configs/detector.yaml`).

The output goes to `data/processed/<name>/plant[_min<size>mm]`, where `<name>` is the config's
`name` (`public7`) or else the dataset names joined by +. A short name keeps run and preview
paths under Windows' 260 characters.

## Only datasets whose every plant is boxed

A detector learns every unboxed area of an image as "not a plant". So a dataset joins the
combination only when every plant in its images has a box or lies in an ignore region. The
species and the outline don't matter, so datasets labelled only crop/weed, or with boxes
only, qualify too. Datasets that box only some plants don't, however many boxes they have:
see [detector.md](detector.md#datasets) for the ones converted and left out. Check a new
dataset with `scripts/preview_annotations.py` before adding it.

## Ignore regions

An annotation has `iscrowd = 1` when it should neither be learned as an object nor be
treated as background:

- plants the source dataset could not identify (e.g. CropAndWeed "Vegetation"),
- labels left unmapped by the chosen taxonomy,
- plants under `min_plant_size_mm` (they keep their own category in `ignored_category_id`),
- partly visible or untrusted plants, where the source marks them (PhenoBench, ACRE),
- boxes whose plant pixels could not be found, and vegetation present in a mask but without
  any box (CropAndWeed).

An ignore region keeps the category of its label when that label has one, and uses the
`ignore` category otherwise (always `ignore` in the combined data). Training neither learns
from ignore regions nor penalises detections on them (`IgnoreAwareRPN`, label -1 in
`weedseg.detector`). **Evaluation:** COCO's standard evaluation ignores crowd regions only for
predictions of the same category, so ignore regions are matched against predictions of any
class, whatever category the region kept (`weedseg.evaluation`).

## Splits

Splits are made by `group` (recording session, field, date), never per image. Neighbouring
frames of one session are near-duplicates and would leak into the test set. The split
fractions and the seed are in the dataset config. A dataset's own split is kept instead when
it is cleaner than ours, e.g. PhenoBench's, which separates crop rows
(`convert_dataset(splits=...)`).
