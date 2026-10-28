# Two-stage pipeline: plant detector + species classifier

An alternative to the species Mask R-CNN ([maskrcnn.md](maskrcnn.md)), scored by the same code
so the two can be compared file by file.

1. **Detector:** the same Mask R-CNN, trained on one class, `plant`. It finds and outlines the
   plants, whatever their species.
2. **Classifier:** DINOv2 ViT-B/14 (timm; Apache-2.0 code and weights) with a linear head,
   fine-tuned on square crops around each plant. The classes of the combined species data,
   `other_plant` included.

Why: the species model's weak point is finding plants, not naming them. A one-class detector
doesn't split a plant's score over similar species (a plant at 0.4 / 0.3 / 0.2 over three
species can fall under the threshold), and it can later learn from datasets that label
plants but not species (PhenoBench, Sugar Beets, WeedsGalore). The classifier sees each
plant at full resolution, starts from a strong pretrained backbone and can learn from
datasets that label species with boxes only (GROUNDBASED_WEED, OPPD).

## Steps

`scripts/run_twostage.py` runs steps 1-7 in order (settings: `configs/run_twostage.yaml`, the
list of steps), e.g. overnight on the training machine. Each step reads its own config;
the runner only orders them, after checking that the configs use the same data (see below).
Re-running is safe: crops already cut for the same data are skipped and trainings resume from `last.pt`
(saved every `train.checkpoint_every` steps; a finished training does nothing).

**Crashes.** Each step runs in a process of its own. When that process crashes, e.g. the
GPU driver resets (Windows does this when the GPU stops responding, and the process dies
without a Python error), the runner waits `restart_wait_s` and starts the step again, up
to `max_restarts` times in all; a training then loses only the steps since its last
checkpoint. A Python error (wrong setting, missing file) stops the runner instead, since
a restart would fail the same way: fix the cause and start it again. Frequent crashes
point at the machine: another program using the GPU, GPU memory near full, or an unstable
card.

**Log.** Everything the runner prints also goes to
`runs/twostage/logs/run_twostage_<date>_<time>.log` (`log_dir`), one file per start, with
the Python lines where a step crashed. Progress bars appear once, in their final state.

**Classifier variants** (`classifier_variants`, added 2026-10-26). A classifier run takes about
an hour, so one session can train several, one after the other, to compare settings. Each
variant is a config holding only what differs from `classifier.yaml` (`base:`), with its own
`run_name`, e.g. `configs/classifier_detail.yaml`. After the steps, each variant repeats the
classifier steps of the list (`prepare_crops`, which skips a cache already cut,
`train_classifier`, `evaluate_classifier`, `predict_twostage` with the same detector) with
its config. Before anything starts, the runner checks that every variant has its own run
folder (a variant that kept the base `run_name` would resume the base's finished run instead
of training anew) and the same species data. A variant that fails with a Python error is
reported and the next one starts; an error in the main steps still stops the runner. At the
end, `compare_<date>_<time>.md` and `.csv` in `log_dir` set all classifier runs side by side
(see [Comparing runs](#training)); `scripts/compare_runs.py` (`configs/compare_runs.yaml`)
does the same for any runs, older ones included. The shipped variants:

| Config | Changes | Why | Result, 2026-10-26/27 |
|---|---|---|---|
| `classifier_seed1.yaml` | `seed: 1` | Run-to-run noise: a change must beat the gap between these two runs | Val accuracy 94.0% against 93.7%: about 0.3 point of chance overall, 1 point on OPPD and in the decision shares, far more for a class with few val plants |
| `classifier_detail.yaml` | `detail_p` 0.5, `blur_p` 0.2, `jpeg_p` 0.3 | [Less detail](#crops) | Worse: 1.5 points lower overall, 4 on OPPD; no cross-dataset gain. Left off |
| `classifier_colour.yaml` | colour jitter doubled (±40%, hue ±0.05) | Camera and light colours (OPPD's greenhouse) tell the dataset less | Within chance everywhere. Not adopted |
| `classifier_frozen.yaml` | `freeze_blocks: 12`: only the final norm and the head learn (a linear probe) | A reference, not a candidate: what fine-tuning adds; data changes checked with less noise; whether a confusion is already in the generic features | Far below: 85.9% overall, balanced 64.2% against 79.8%, cross-dataset 70.5% against 87.2%. Fine-tuning adds 8 points, too much for it to stand in for the real model as a benchmark. ACRE's beans called fat-hen are already in the generic features (7.1% of them removed, as in the fine-tuned runs) |
| `classifier_dinov3.yaml` | DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`) | Newer backbone. Its weights are under Meta's DINOv3 License, not Apache-2.0: a copy of the licence must go with the model and "Built with DINOv3" be shown; adopting it for the product is a licensing decision | Best so far, one run: 94.4% overall, balanced 81.3% against 79.8% for both DINOv2 runs, OPPD balanced 90.1% against 85%, CropAndWeed 91.8% against 90.9%; decisions as DINOv2's. Partial labels fit a little less (92.7% against 94.8%), a share that moves by several points between epochs. About as fast (5:15 against 5:43 per epoch: data loading sets the pace). A second seed should confirm it before a switch |
| `classifier_balance.yaml` | `train.balance: sqrt` | ACRE leans towards fat-hen (83% of its plants), and 6% of its beans would be removed | 93.2% overall (0.5 to 0.8 point lower), balanced 80.6% (0.8 higher), OPPD 1 to 2 lower. ACRE's beans removed fall to 2.3%, but its harmful weeds removed fall from 52% to 30% too: the model is less sure, so more is deferred. At the same share of beans removed, it removes as many weeds as the other runs (see below). Not adopted |
| `classifier_seed2.yaml` | `seed: 2` | A third DINOv2 seed: each backbone judged against the spread of its own three runs | Pending |
| `classifier_dinov3_seed1.yaml`, `classifier_dinov3_seed2.yaml` | DINOv3, `seed` 1 and 2 | Whether DINOv3's lead is beyond chance, before a switch | Pending |
| `classifier_dinov3_frozen.yaml` | DINOv3 with `freeze_blocks: 12` | DINOv3's generic features next to DINOv2's (frozen: 85.9%), and what fine-tuning adds to each | Pending |

The first three ran on 2026-10-26, the next three on 2026-10-27, the last four are for the
next session. Results from the last epoch (the first three evaluated again on 2026-10-27). A
per-dataset balanced accuracy can jump on a single plant: ACRE's val crops hold 4 maize plants, and getting one of them right
raised the detail run's ACRE balanced accuracy by 3 points.

**Decisions at one `defer_below` compare confidence, not only skill.** With label
smoothing 0.1, a class gets at most about 0.9, so the summed probability of an action
rarely passes 0.92 to 0.94, and `defer_below` 0.9 sits just under that. A model a little
less sure (like the sqrt run) then defers far more of everything: fewer crops removed, but
fewer weeds too. To compare decisions between runs, compare them at the same share of crops
removed. A sweep of `defer_below` on the val predictions (2026-10-28) puts the base, seed1,
DINOv3 and sqrt runs on one curve for ACRE: about 52% of its harmful weeds removed with 6.3%
of its beans, 43 to 45% with 4.2%, 31 to 33% with 2.5%, 18 to 23% with 1.6%. No variant moves
the curve; only fewer beans called fat-hen would. The sweep also shows how narrow the useful
range is: for the base run, ACRE's beans removed go from 4.2% at 0.922 to 0.3% at 0.932.

| Step | Script | Config | Where |
|---|---|---|---|
| 1. Plant-only copy of the species data | `make_plant_dataset.py` | `plant_dataset.yaml` | any machine |
| 2. Crop cache | `prepare_crops.py` | `classifier.yaml` | any machine (CPU) |
| 3. Train the detector | `train_detector.py` | `detector.yaml` | GPU |
| 4. Evaluate the detector alone | `evaluate_detector.py` | `detector.yaml` | GPU |
| 5. Train the classifier | `train_classifier.py` | `classifier.yaml` | GPU |
| 6. Evaluate the classifier alone | `evaluate_classifier.py` | `classifier.yaml` | GPU |
| 7. Predict and evaluate both stages | `predict_twostage.py` | `twostage.yaml` | GPU |
| 8. Review the errors | `review_errors.py` | `maskrcnn.yaml`, `review.eval_dir` = the folder of step 7 | any machine |

Steps 1 and 2 need no GPU: they can run before a session on the training machine (copy the
`crops_*` folder along if the data is copied rather than re-made there). The DINOv2 weights
(~350 MB) are downloaded from Hugging Face on the first run; for an offline machine set
`model.pretrained_path` to the downloaded `model.safetensors`.

**Configs.** `detector.yaml` holds only what differs from `maskrcnn.yaml` (`base:`): the
plant-only `data.dir`. Every other setting comes from `maskrcnn.yaml`, so the detector and the
species model are always trained alike, and `maskrcnn.yaml` stays on the species model.
`twostage.yaml` names the detector and classifier configs and takes `best` or `last` of
their runs (the classifier's `last`, see [Training](#training)), so the prediction uses what
was just trained. The runner checks that
`plant_dataset.yaml`, `classifier.yaml` and `twostage.yaml` use the same species `data.dir`
and `detector.yaml` the plant folder made from it.

Step 4 shows whether the one-class detector finds more plants than the species model (AP and
recall); taxonomic levels and decisions are skipped there, since `plant` is not a species.

## Plant-only data

Every labelled plant becomes `plant`. Ignore regions stay ignored (plants under the minimum
size, unidentified vegetation, plants whose mask the converter couldn't derive), except
plants hidden only by the choice of classes: a genus-only label that combining turned into an
ignore region because one of its species is a class (rule 3 in
[data-format.md](data-format.md)). To a detector that is a plant like any other. On
CropAndWeed + ACRE that gives back 474 train plants. The same goes for plants of unknown
species (`1PLAK`): datasets that label each plant but not its species, e.g. ACRE's
"unknown" plants or PhenoBench's weeds, add to the detector's training data only.
Combining marks both kinds `hidden_plant`, so plants the converter left out (untrusted, no
usable outline) stay ignored even when they carry a code. See `weedseg.data.plant`.

## Datasets labelled with boxes only

All datasets go into one combination with one class list, the classifier's, so the
evaluation measures the classes the robot will use. Each stage takes what the labels allow
(images marked `box_only`, see [data-format.md](data-format.md#datasets-labelled-with-boxes-only)).
For the datasets planned on 2026-10-21:

| | CropAndWeed, ACRE (masks, species) | PhenoBench (masks; weeds without species) | GROUNDBASED_WEED, OPPD (boxes, species) |
|---|---|---|---|
| Detector | yes | every plant | no |
| Classifier | yes | sugar beets; weeds as "not sugar beet" ([partial labels](#partial-labels)) | yes |
| Evaluation of both stages (step 7) | yes | sugar beets; step 4 also counts its weeds | not yet |
| Evaluation of the classifier (step 6) | yes | sugar beets; weeds by `label_fit` | yes |

- The species Mask R-CNN can't learn from box-only datasets: it stays the reference trained
  on CropAndWeed + ACRE ([maskrcnn.md](maskrcnn.md)). Compare it with the two-stage pipeline
  in the CropAndWeed and ACRE rows of `by_dataset.csv`. Its plants found stay comparable;
  its class numbers drift a little once a new dataset turns some of its `other_plant`
  species into classes.
- A dataset that changes the class list needs a new classifier (~30 min); the detector is
  retrained only when a dataset with masks joins. The "all" row of `by_dataset.csv` changes
  with the datasets, so compare runs row by row.
- On box-only datasets the classifier is scored on the labelled plants' crops only, not on
  detections. Scoring detection there too (matching by box, box AP only) is planned, to see
  whether the detector finds plants in fields it never saw.

## Crops

A plant's crop is a square centred on its box, with `crops.margin` (25%) of the box's longer
side as context on each side, resized to `crops.input_size` (224 px). Crops at image edges are
padded with the ImageNet mean grey.

**Cache.** `prepare_crops.py` cuts every labelled plant once, and every plant with a partial
label, 1.5 times larger than the crop (`cache_context`, room for box jitter), with its mask:
`<data.dir>/crops_margin<m>_size<s>_context<c>/<split>/<annotation id>.jpg|.png` plus
`index_<split>.csv` (class, EPPO code, size in mm, mm/px, dataset, and the allowed classes of a
partial label). Changing margin, size or context makes a new cache folder next to the old
one. A split whose index no longer matches the data (re-combined, a dataset added or
re-converted) is cut again; one that matches is skipped.

**Training augmentation:** box jitter (crop side ±20%, centre ±10% of the box), so a model
trained on labelled boxes copes with the detector's; flips and 90° rotations (top-down images
have no "up"); colour jitter.

**Less detail** (`augment.detail_p`, `blur_p`, `jpeg_p`; off by default, added 2026-10-26).
Every crop is resized to 224 px, so the classifier doesn't see a plant's size, but it does
see how much detail the source camera caught: a 40 mm plant's crop spans ~400 px in OPPD
(0.15 mm/px, capped at 224 by the resizing), 146 in CropAndWeed, 120 in ACRE and 60 in
PhenoBench (1 mm/px). Sharpness thus tells the dataset, and a species learnt from sharp
crops may not be recognised in coarse ones. `detail_p` is the share of training crops shrunk
to the detail of a coarser camera, a random mm/px between their own and `max_mm_per_px`
(1.2), and enlarged back; `blur_p` and `jpeg_p` add Gaussian blur and JPEG compression. The
crop's own detail is the coarser of the source's mm/px (index column `mm_per_px`) and what
224 px can hold. Like any augmentation it can also cost accuracy, since fine detail is a
real species cue. **Tried on 2026-10-26** (`classifier_detail.yaml`: 0.5, 0.2, 0.3), it did:
val accuracy 92.5% against 94.0% without, 87.6% against 91.7% on OPPD, whose tiny seedlings
need the detail, and no gain on species learnt from other datasets (cross-dataset accuracy
86.4% against 87.2%). So it stays off; a much milder setting could still be tried, but
these numbers give little reason to expect a gain.
Resampling all images to one fixed mm/px instead would throw away OPPD's detail and build
the estimated mm/px of three datasets into the data.

**Background** (`crops.background`). Neighbouring plants can share a crop, e.g. a small weed
next to maize. `keep` (default) leaves the crop as it is; the classifier learns that the label
belongs to the plant in the centre. `dim` darkens the pixels outside the plant's mask to
`dim_factor` (30%): context stays visible, the target is obvious. `blank` greys them out, but
also loses leaf tips the mask misses (the detector's masks are rougher than the labels).
Plants labelled with boxes only have their box as mask, so these two treat them less well. With
`dim` or `blank`, training masks are grown or shrunk by up to `mask_jitter_px` to mimic the
detector. The cache always stores masks, so switching needs no new cache, only a new
classifier run.

## Training

AdamW with a low learning rate for the pretrained backbone (2e-5) and a high one for the new
head (1e-3), warm-up and cosine decay, label smoothing 0.1, stochastic depth 0.1, mixed
precision. `train.balance` can draw rare classes more often (`sqrt` or `inverse`); default
`none`. Two checkpoints are kept: `last.pt`, the final epoch (runs also resume from it, like
the Mask R-CNN), and `best.pt`, the epoch with the best val accuracy. Outputs:
`runs/classifier/<run_name>/`.

**Which checkpoint.** Evaluation, prediction and the export use `last` (`evaluate.checkpoint`
in `classifier.yaml`, `classifier.checkpoint` in `twostage.yaml`) since 2026-10-27. By the
last epoch the learning rate has decayed to almost zero, and in the four runs of 2026-10-26
it was within 0.15 point of the best epoch; `best` only picked chance peaks. Once that cost
a lot: one run kept epoch 4, 0.1 point above its last, and that half-trained model was less
sure of itself, so it removed 30% of the harmful weeds where the others removed about 50%
(the decisions act only on probabilities of at least `defer_below`, 0.9, about where label
smoothing caps them). `best` would matter if a longer training overfitted: training then ends
with a warning when the last epoch is more than `train.warn_last_below_best` (half a point)
below the best, and `best.pt` can be evaluated too. Each evaluation goes into a folder of its
checkpoint (`eval/val_last`, `eval/val_best`, also for both stages), so they don't overwrite
each other.

`evaluate_classifier.py` scores the classifier on the labelled crops of a split, without the
detector: accuracy and balanced accuracy, per class, confusion matrix, `taxonomic_levels.csv`
(compare with the species model's, which also counts only plants it found), accuracy by plant
size (`accuracy_by_size.csv`, to see whether small plants suffer from their neighbours),
accuracy per source dataset (`accuracy_by_dataset.csv`, for a split that combines several)
and `predictions.csv`. Plants with partial labels are left out of all of these; they have
their own files (next section).

`accuracy_by_class_and_dataset.csv` (for a split that combines several datasets) gives the
accuracy of each class in each dataset, for the classes found in more than one, with the
share of the class's training crops that came from that dataset and its most frequent wrong
answer. It shows how well a species learnt mostly from one dataset is recognised in another,
e.g. run 2 recognised none of 18 CropAndWeed black-bindweeds (`POLCO`), learnt mostly from
OPPD. `cross_dataset_accuracy` in `metrics.json` is the accuracy over such crops (under
`evaluate.cross_dataset_share`, 50%, of the class's training crops from their own dataset;
`other_plant` left out, since it holds different species in each dataset).

**Comparing runs.** The aim is a better classifier overall. Judge a change first on
accuracy and balanced accuracy, per dataset (`accuracy_by_dataset.csv`), and on the
decisions of both stages (`by_dataset.csv` of step 7); then on `cross_dataset_accuracy` and
the partial labels' fit, which show whether it relies less on the dataset's look. A change
that raises the latter while the former drop is not an improvement by itself. A gap is only
worth something if it is larger than the gap between two runs that differ by their seed
alone (`classifier_seed1.yaml`).

`weedseg.compare` lays this out (after classifier variants, or with `scripts/compare_runs.py`):
one column per run (A, B, ... named above the table), and rows in that order: the settings in
which the runs differ (from each run's `config.yaml`), the epoch evaluated, how far the last
epoch's val accuracy is below the best's, the classifier's accuracy overall and per dataset, both stages' plants found, species right, crops removed and
harmful weeds removed and left, per dataset, then cross-dataset accuracy and the partial
labels' fit. Runs that differ only in `seed` are pointed out. The two-stage numbers come from
the newest `runs/twostage/<detector>__<classifier run>/eval/<split>_<checkpoint>/`, made with
the same checkpoint (folders named `<split>` were made up to 2026-10-26, with `best`).

### Partial labels

Up to run 2 (2026-10-23) the classifier saw only plants labelled with their class. That
taught it a shortcut: PhenoBench labels only sugar beet with a species (its weeds are
`1PLAK` ignore regions), so every PhenoBench-like crop it saw was a beet, and it called
**all** 3,090 PhenoBench val weeds sugar beet, with the median probability of the real
beets (0.91). Images that look like PhenoBench's would get the same answer in the field.

Since 2026-10-26 (`data.partial_labels: true`), plants hidden by the choice of classes train
the classifier with what their label still says ([data-format.md](data-format.md#partial-labels)):
a PhenoBench weed is anything but sugar beet, a CropAndWeed *Poa* is `POAAN` or
`other_plant`. The loss is minus the log of the probability the classifier gives to all the
allowed classes together (`PartialLabelLoss`), which for one allowed class is the usual
cross-entropy. So the classifier is pushed off the forbidden classes without being told
which allowed class is right. Label smoothing applies to these plants as to the others.
`false` leaves them out, as before.

On the five datasets (2026-10-26, 68 species classes), the partial labels add 12,413
training plants and 4,916 val plants (plants of at least 20 mm, val in brackets):

| Dataset, label | Allowed classes | Train (val) |
|---|---|---|
| PhenoBench weeds | every class but `BEAVA` | 5,967 (3,090) |
| OPPD dicots of unknown species (`PPPDD`) | every class but the 14 monocot ones (grasses, `CYPRO`, `ZEAMX`) | 2,732 (926) |
| CropAndWeed *Poa* (genus only) | `POAAN`, `other_plant` | 1,645 (254) |
| OPPD monocots of unknown species (`PPPMM`) | the 14 monocot classes, `other_plant` | 1,130 (295) |
| CropAndWeed *Geranium* | `GERMO`, `GERPU`, `other_plant` | 438 (24) |
| ACRE plants without a name, class "weed" | every class but ACRE's crops (`PHSVX`, `ZEAMX`, `VICFX`, `PIBSX`) | 361 (216) |
| OPPD Asteraceae (family only) | the 10 Asteraceae classes, `other_plant` | 95 (71) |
| ACRE plants without a name, class "crop" | `PHSVX`, `ZEAMX`, `VICFX`, `PIBSX` | 16 (7) |
| CropAndWeed *Veronica*, *Sisymbrium*; ACRE *Digitaria*, *Setaria* | the genus's classes, `other_plant` | 29 (8) |
| CropAndWeed crop poppy (*Papaver*) | `other_plant` (a crop, so not the weed `PAPRH`) | 0 (25) |

The CropAndWeed *Poa* and *Geranium* plants trained as `other_plant` in run 1
(CropAndWeed+ACRE); OPPD made `POAAN` and `GERMO` classes, which turned them into ignore
regions (rule 3), so run 2 learnt *Poa annua* almost only from OPPD's greenhouse trays.
Plants whose label allows every class stay out: GROUNDBASED_WEED's `NR` (472 training
plants) and ACRE's "unknow" plants.

**Scoring.** These plants can't be right or wrong, but the predicted class can fit the label
or not. The share that fits is `val_partial_label_fit` in `metrics.jsonl` during training,
and `partial_label_fit` in the evaluation's `metrics.json`. `partial_labels.csv` gives it per
dataset and label, with the classes most often predicted (run 2: 0% of the PhenoBench weeds
fit, 100% called `BEAVA`; with partial labels, 2026-10-26: about 95% fit, while the
PhenoBench beets stayed 99% right and no dataset shared with run 2 got worse); `partial_predictions.csv` has one row per crop. The export's
README shows the same table. They don't decide which checkpoint is used.

What this doesn't fix: species seen in a single dataset can still be told apart by the
dataset's look rather than the plant, e.g. *Setaria verticillata* (GROUNDBASED_WEED only)
and *Setaria viridis* (CropAndWeed only). Only more sources of those species help there;
`accuracy_by_dataset.csv` and the confusion matrix show it.

## Prediction

`predict_twostage.py` runs the detector on the split, classifies every detection with a plant
score of at least `classifier.min_plant_score`, and writes the same outputs as
`evaluate_maskrcnn.py` to
`runs/twostage/<detector run>__<classifier run>/eval/<split>_<classifier checkpoint>/` (e.g.
`val_last`; up to 2026-10-26 `<split>/`, always made with `best`).

Each detection answers two questions separately: `score` (and `plant_score`) is the
detector's plant score, how sure it is that there is a plant; `class_scores[c]` is the
classifier's probability of `c`, given that it is a plant, and the class is the most probable
one. The evaluation's score threshold and AP use the plant score; the decision rules use
each for its own threshold (`plant_defer_below` and `defer_below`, see
[maskrcnn.md](maskrcnn.md#decisions)). The species model has the same two numbers, only
mixed: its class probabilities sum to its plant probability.

They were first multiplied (`score` = plant score × classifier probability), to copy the
species model, but on the first run (CropAndWeed+ACRE, 2026-10-21) that worked badly:

- Label smoothing keeps the classifier's probabilities under ~0.9, so the product rarely
  reached `defer_below` 0.9: 43% of harmful weeds were deferred.
- The species is equally often right (94-95%) at every plant score. A low plant score means
  "maybe not a plant" (half the detections at 0.5-0.8 lie on no labelled plant), not "maybe
  the wrong species", so it should not lower the confidence in the action.
- The product also pushed plants under the 0.5 detection threshold (73% of plants found
  instead of 78%), and ranking by the plant score alone gives a slightly higher AP
  (CropAndWeed images: segm 0.376 instead of 0.364).

One difference in AP: the species model can return the same plant twice with two classes
(its post-processing makes one detection per proposal and class), which can add recall at
low scores; the two-stage pipeline returns each plant once, with its best class. The
confusion matrix, taxonomic levels and decisions, at score 0.5, are unaffected.

## Deployment package

`scripts/export_twostage.py` (settings: `configs/export_twostage.yaml`) packages the
detector and classifier of `configs/twostage.yaml` for someone who runs the model outside
this repo, e.g. on a demo website. It writes `exports/weed_twostage_<date>/`:

| File | |
|---|---|
| `detector.safetensors`, `classifier.safetensors` | The weights only: plain tensors, no pickled code |
| `model.json` | What `weed_model.py` needs (image size, anchors, crop margin, normalisation), the classes with their scientific and English names, the training datasets and their licences, and the hand-off thresholds |
| `weed_model.py` | Self-contained inference, `WeedModel.predict(rgb_image)`: torch, torchvision, timm, safetensors, numpy and opencv, nothing from `weedseg` |
| `requirements.txt` | Those packages |
| `README.md` | For the person deploying it: install, input, output, hand-off, the species list, validation numbers, training data and licences |

The README is made from `src/weedseg/deploy/README_template.md`, with the species,
training data and validation numbers of the runs read from their evaluation folders (run
steps 6 and 7 first). `weed_model.py` repeats the preprocessing of `predict_twostage.py`
(scaling, crop geometry, padding, normalisation) bit for bit. `tests/test_deploy.py` checks
it against the training code, so a change there that the package misses makes the tests
fail.

Each plant comes out with its box, outline (polygons), plant score, species, top 5 and a
`send_to_zero_shot` flag for the generic model: set when the top class is `other_plant`, the
plant score is under `zero_shot_below_plant_score` (0.8, as `plant_defer_below`) or the
species probability is under `zero_shot_below_species_probability` (0.7: on the val crops of
2026-10-23, 90% of the plants stay with the model, 97% of them right). No roles or
keep/remove decisions: those depend on the field's rules.

On 2026-10-23 the package gave the same plants, species and masks as `predict_twostage.py`
on CropAndWeed, ACRE and PhenoBench val images (CPU, full precision). The only differences
were the package's rounding of boxes and scores.

## Inference on any images

`scripts/demo_twostage.py` (settings: `configs/demo_twostage.yaml`) runs an exported
package on images with no labels: one image, a folder or a list of both. It writes to
`runs/demo/<package>/`, per image, `<name>.jpg` with the plants drawn on it (mask and
outline coloured by species, label with the species probability; plants sent to the zero-shot
model in white, labelled "zero-shot") and `<name>.json` with the plants as the package's
README describes them. It prints the time and the species found per image; `open_results`
opens each drawing.

It runs the package's own `weed_model.py`, so it shows what the package's user gets. It
neither scores nor needs labels: for numbers on a labelled split, use `predict_twostage.py`.
On a laptop CPU it takes about 10 s per image (2026-10-25: 10.4 s for a CropAndWeed image
with 8 plants), plus 3 s to load the model.
