# Two-stage pipeline: plant detector + species classifier

An alternative to the species Mask R-CNN ([maskrcnn.md](maskrcnn.md)), scored by the same code
so the two can be compared file by file.

1. **Detector:** the same Mask R-CNN, trained on one class, `plant`. It finds and outlines the
   plants, whatever their species.
2. **Classifier:** DINOv3 ViT-B/16 (timm, Apache-2.0 code; weights under Meta's DINOv3
   License) with a linear head, fine-tuned on square crops around each plant. The classes of
   the combined species data, `other_plant` included. DINOv2 ViT-B/14 (Apache-2.0 weights)
   until 2026-10-28, now `classifier_dinov2.yaml`: DINOv3 was better in three seeds each (see
   [Steps](#steps)).

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
of training anew) and the same species data, unless it names its own `data.dir`: a variant
may train on another combination of `configs/combine.yaml` (since 2026-10-30, e.g.
`classifier_imageweeds.yaml`), and its prediction then runs on that combination's split, with
the same detector. The species data of every run must exist. A variant that fails with a Python error is
reported and the next one starts; an error in the main steps still stops the runner. At the
end, `compare_<date>_<time>.md` and `.csv` in `log_dir` set all classifier runs side by side
(see [Comparing runs](#training)); `scripts/compare_runs.py` (`configs/compare_runs.yaml`)
does the same for any runs, older ones included. The shipped variants:

| Config | Changes | Why | Result, 2026-10-26 to 28 |
|---|---|---|---|
| `classifier_seed1.yaml` | `seed: 1` | Run-to-run noise: a change must beat the gap between these two runs | Val accuracy 94.0% against 93.7%: about 0.3 point of chance overall, 1 point on OPPD and in the decision shares, far more for a class with few val plants |
| `classifier_detail.yaml` | `detail_p` 0.5, `blur_p` 0.2, `jpeg_p` 0.3 | [Less detail](#crops) | Worse: 1.5 points lower overall, 4 on OPPD; no cross-dataset gain. Left off |
| `classifier_colour.yaml` | colour jitter doubled (±40%, hue ±0.05) | Camera and light colours (OPPD's greenhouse) tell the dataset less | Within chance everywhere. Not adopted |
| `classifier_frozen.yaml` | `freeze_blocks: 12`: only the final norm and the head learn (a linear probe) | A reference, not a candidate: what fine-tuning adds; data changes checked with less noise; whether a confusion is already in the generic features | Far below: 85.9% overall, balanced 64.2% against 79.8%, cross-dataset 70.5% against 87.2%. Fine-tuning adds 8 points, too much for it to stand in for the real model as a benchmark. ACRE's beans called fat-hen are already in the generic features (7.1% of them removed, as in the fine-tuned runs) |
| `classifier_dinov3.yaml` (the base `classifier.yaml` since 2026-10-28) | DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`) | Newer backbone. Its weights are under Meta's DINOv3 License, not Apache-2.0: a copy of the licence must go with the model and "Built with DINOv3" be shown; adopting it for the product is a licensing decision | Better, confirmed by three seeds (with `classifier_dinov3_seed1/2.yaml`): see below. About as fast (58 against 63 min per training: data loading sets the pace) |
| `classifier_balance.yaml` | `train.balance: sqrt` | ACRE leans towards fat-hen (83% of its plants), and 6% of its beans would be removed | 93.2% overall (0.5 to 0.8 point lower), balanced 80.6% (0.8 higher), OPPD 1 to 2 lower. ACRE's beans removed fall to 2.3%, but its harmful weeds removed fall from 52% to 30% too: the model is less sure, so more is deferred. At the same share of beans removed, it removes as many weeds as the other runs (see below). Not adopted |
| `classifier_seed2.yaml` | `seed: 2` | A third DINOv2 seed: each backbone judged against the spread of its own three runs | 93.7%, as the base. Over the three DINOv2 seeds, chance moves overall and balanced accuracy by 0.3 point, OPPD's balanced by 1.7, and both stages' CropAndWeed harmful removed by 1.6 |
| `classifier_dinov3_seed1.yaml`, `classifier_dinov3_seed2.yaml` | DINOv3, `seed` 1 and 2 | Whether DINOv3's lead is beyond chance, before a switch | It is: see below |
| `classifier_dinov3_frozen.yaml` | DINOv3 with `freeze_blocks: 12` | DINOv3's generic features next to DINOv2's (frozen: 85.9%), and what fine-tuning adds to each | 83.0% overall, balanced 57.8%, OPPD 71.0% (frozen DINOv2: 85.9%, 64.2%, 79.0%). Frozen, DINOv3 is the worse backbone; fine-tuned, the better (fine-tuning adds 11 points against 8). So a frozen run can't even rank backbones, let alone stand in for the real model |

The first three ran on 2026-10-26, the next three on 2026-10-27, the last four on 2026-10-28.
Results from the last epoch (the first three evaluated again on 2026-10-27). A
per-dataset balanced accuracy can jump on a single plant: ACRE's val crops hold 4 maize plants, and getting one of them right
raised the detail run's ACRE balanced accuracy by 3 points.

**DINOv2 against DINOv3, three fine-tuned seeds each** (2026-10-28, val, last epoch; each
cell gives the three seeds):

| | DINOv2 | DINOv3 |
|---|---|---|
| Accuracy | 93.7 / 94.0 / 93.7 | 94.4 / 94.1 / 94.1 |
| Balanced accuracy | 79.8 / 79.8 / 79.5 | 81.3 / 81.3 / 81.0 |
| OPPD balanced | 85.1 / 84.9 / 86.6 | 90.1 / 88.7 / 88.2 |
| CropAndWeed accuracy | 90.9 / 90.8 / 90.7 | 91.8 / 91.7 / 91.6 |
| CropAndWeed balanced | 71.0 / 71.2 / 70.5 | 72.1 / 72.5 / 73.1 |
| Both stages, CropAndWeed species right | 93.2 / 93.5 / 93.3 | 94.2 / 93.9 / 93.9 |
| Both stages, harmful removed with up to 1% of crops, all | 49.5 / 51.3 / 51.0 | 51.8 / 52.4 / 52.1 |
| Cross-dataset accuracy | 87.2 / 87.2 / 87.1 | 87.4 / 87.4 / 87.6 |
| Partial labels fit | 94.8 / 94.8 / 94.9 | 92.7 / 93.5 / 93.2 |
| Partial labels fit, PhenoBench | 94.9 / 94.7 / 94.7 | 91.8 / 92.8 / 92.5 |

In every row each DINOv3 seed beats each DINOv2 seed, except the partial labels, where each
is worse. The gain is general (all of OPPD, CropAndWeed and the hard classes), not a
shortcut fix; ACRE and GROUNDBASED_WEED stay the same. ACRE's
decisions too: both backbones remove as many of ACRE's weeds for a given share of its beans
(e.g. 47 to 49% with up to 5%), since its beans called fat-hen limit both. Against that,
DINOv3 calls about 7.5% of PhenoBench's unnamed weeds sugar beet (DINOv2: 5%) and fits
CropAndWeed's genus labels 3 points less. So `classifier.yaml` switched to DINOv3 on
2026-10-28 (the DINOv3 seed-0 run is its run), and DINOv2 became `classifier_dinov2.yaml`,
which the DINOv2 variants now build on. The export ships the DINOv3 License with the weights
and says "Built with DINOv3" (see [Deployment package](#deployment-package)).

**Adding Weed-AI datasets** (prepared 2026-10-30; [datasets/weedai.md](datasets/weedai.md)).
The main combination became `public7`: the five datasets plus VCD and LincolnBeet, 69
species classes (leek, `ALLPO`, is new), 16,709 training images against 12,233. ImageWeeds
has potted weeds ([datasets/imageweeds.md](datasets/imageweeds.md)), so it is a variant
first: `public7+imageweeds`, 73 classes (common ragweed, kochia, waterhemp, horseweed). All
four new datasets have boxes only, so the detector is unchanged (run 2's) and only the
classifier is trained, two seeds per combination:

| Run | Config | Data |
|---|---|---|
| A, B, C | the 2026-10-28 DINOv3 runs (seeds 0-2), as reference | five datasets |
| D, E | `classifier.yaml`, `classifier_dinov3_seed1.yaml` | `public7` |
| F, G | `classifier_imageweeds.yaml`, `classifier_imageweeds_seed1.yaml` | `public7+imageweeds` |

Each combination's val split holds the previous one's val images with the same boxes and
classes (checked: all 4,248 of `public7` in `public7+imageweeds`), so compare runs dataset
by dataset, not by the overall row, whose mix of datasets differs. The questions:

- **D, E against A-C**: do VCD and LincolnBeet hurt the five datasets' rows? Do they help
  ACRE's beans (VCD is from ACRE's site, [provenance](datasets/provenance.md)) and
  PhenoBench's "not sugar beet" fit?
- **F, G against D, E**: does ImageWeeds hurt the seven datasets' rows? Redroot pigweed
  (`AMARE`) on CropAndWeed is the check of the pots: ImageWeeds is its only other source.
- ImageWeeds's own accuracy (rows of F and G) is only a weak sign: its val split shares the
  three sites and the pots.

Results: pending.

**Decisions at one `defer_below` compare confidence, not only skill.** With label
smoothing 0.1, a class gets at most about 0.9, so the summed probability of an action
rarely passes 0.92 to 0.94, and `defer_below` 0.9 sits just under that. A model a little
less sure (like the sqrt run) then defers far more of everything: fewer crops removed, but
fewer weeds too. To compare decisions between runs, compare them at the same share of crops
removed: the comparison does it since 2026-10-28 (`crops_removed` in
`configs/compare_runs.yaml`, see [Comparing runs](#training)). A sweep of `defer_below` on the
val predictions (2026-10-28) puts the base, seed1,
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
`crops_*` folder along if the data is copied rather than re-made there). The DINOv3 weights
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
For the datasets planned on 2026-10-21, and the Weed-AI ones of 2026-10-30:

| | CropAndWeed, ACRE (masks, species) | PhenoBench (masks; weeds without species) | GROUNDBASED_WEED, OPPD, VCD, ImageWeeds (boxes, species) | LincolnBeet (boxes; weeds without species) |
|---|---|---|---|---|
| Detector | yes | every plant | no | no |
| Classifier | yes | sugar beets; weeds as "not sugar beet" ([partial labels](#partial-labels)) | yes | as PhenoBench |
| Evaluation of both stages (step 7) | yes | sugar beets; step 4 also counts its weeds | not yet | not yet |
| Evaluation of the classifier (step 6) | yes | sugar beets; weeds by `label_fit` | yes | as PhenoBench |

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
alone (`classifier_seed1.yaml`, `classifier_seed2.yaml`: see the spread of the three seeds
under Classifier variants in [Steps](#steps)).

`weedseg.compare` lays this out (after classifier variants, or with `scripts/compare_runs.py`):
one column per run (A, B, ... named above the table), and rows in that order: the settings in
which the runs differ (from each run's `config.yaml`), the epoch evaluated, how far the last
epoch's val accuracy is below the best's, the classifier's accuracy overall and per dataset, both stages' plants found, species right, crops removed and
harmful weeds removed and left, per dataset, then the decisions at the same share of crops
removed, then cross-dataset accuracy and the partial labels' fit. Runs that differ only in
`seed` are pointed out. The two-stage numbers come from the newest
`runs/twostage/<detector>__<classifier run>/eval/<split>_<checkpoint>/`, made with the same
checkpoint (folders named `<split>` were made up to 2026-10-26, with `best`).

**Decisions at the same share of crops removed** (`crops_removed` in
`configs/compare_runs.yaml`, e.g. 1%, 2% and 5%; the runner's comparison takes them from
there too). The rule file's one `defer_below` puts each run at a different crop safety, so
for each share and dataset the comparison finds the lowest `defer_below` that removes no
more of that dataset's crops, and shows the share of its harmful weeds removed there, with
that `defer_below` in brackets, e.g. `28.1% (0.927)`. The `defer_below` is rounded up to 3
decimals, so a rule file set to it gives these numbers, and is never under 0.5 (an action
less likely than not). At 0.500 the share was no constraint: even then fewer crops are
removed. The numbers are made again from each two-stage evaluation's `predictions.json` and
the ground truth and settings in its `evaluation.json` (the rule file as it is now, so all
runs get the same roles); the split file must be on the machine, else a note says why the
run has none. This takes about 10 s per run.

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

## Species lists

`weedseg.skew.skew_probabilities` skews the classifier's probabilities with the species
expected in a field: one crop list and one weed list. It has only two settings, `use_lists`
(default off) and `multiplier` (default 5), and doesn't know where the lists come from: an
expert, a first pass of the model, a regional list. Whatever supplies them also answers for
their quality, since a bad list is applied like a good one. The export's `weed_model.py` doesn't
use it. The export's `model.json` tags each class `crop`, and the platform has its own copy of the
step (R69 in its docs/requests.md).

A list makes the species off it less likely, so the model can still name them:

1. Off-list weeds are divided by the multiplier, then all weeds are scaled back to their total.
   The list says which weeds to expect, not whether the plant is a weed, so it can't push a
   plant towards the crop.
2. Off-list crops are divided by the multiplier, and that probability goes to all other classes:
   a crop that isn't in the field doesn't explain the plant.

An empty list says nothing, and that side stays as it was, so a crop list alone is a safe way
to use a field's known crop. A class on either list is never divided and the lists override the
crop tags (volunteer potatoes on the weed list), and `other_plant` is never divided.

`scripts/evaluate_skew.py` (settings: `configs/evaluate_skew.yaml`) measures this from saved
two-stage predictions, in about 40 s without a GPU, and writes `runs/skew/skew_<date>_<time>.md`
and `.csv`, with `_missed.csv` and `_lists.csv` (every session's lists). Without a skew its
species right equals `by_dataset.csv` exactly. Its crops removed and harmful removed count the
plants found, where `by_dataset.csv` also counts those missed, so they run higher: CropAndWeed's
harmful removed is 64.8% here and 49.9% there. No real list exists yet, so it uses stand-ins per
session (image `group`):

- `labelled`: the species labelled in the session. The best case: nothing is missing, and
  sessions label few species.
- `first_pass`: from the raw predictions, the most frequent crop class and the weeds making up at
  least 10% of the plants called weeds, in sessions with at least 40 plants: 60 of CropAndWeed's
  127 val sessions, with a median of 2 weeds listed. The minimum belongs to this source, not to
  the skew step: an expert's list holds for a small session too.
- `first_pass_paired`: the same, with sessions under 40 plants first joined two by two, those
  with the same most frequent crop class, the largest first. For testing only: it reaches 88-90
  sessions, but joins different sites.

Results on val (2026-11-03, the two public7 DINOv3 seeds, x5), of the plants found. Species right
leaves out `other_plant`, as `by_dataset.csv` does. Harmful removed is at up to 2% of crops removed:

| Lists | CnW species right | CnW balanced | CnW crop or not right | CnW weeds called crop | CnW harmful removed | ACRE balanced | ACRE crops removed at 0.9 |
|---|---|---|---|---|---|---|---|
| No skew | 93.8-93.9% | 86.6-86.7% | 98.4% | 1.8-2.0% | 89.6-89.8% | 53.0-53.9% | 6.7-6.8% |
| Labelled, both | 95.9-96.2% | 90.8-91.8% | 98.7% | 1.1-1.2% | 91.2-91.6% | 52.8-53.8% | 7.5-7.8% |
| Labelled, crop only | 93.9-94.1% | 86.8-87.0% | 98.5-98.6% | 1.3-1.6% | 90.0-90.3% | 53.0-53.9% | 7.0-7.1% |
| Labelled, weed only | 95.7-95.9% | 90.6-91.4% | 98.5% | 1.5% | 90.8-91.1% | 52.8-53.8% | 7.3-7.4% |
| First pass, both | 94.4-94.7% | 88.1-88.7% | 98.7% | 1.0-1.2% | 91.3% | 46.5-48.3% | 8.6-9.0% |
| First pass, crop only | 94.0-94.2% | 86.8-87.1% | 98.7% | 1.2-1.5% | 90.2-90.6% | 53.0-53.9% | 7.1% |
| First pass, weed only | 94.2-94.4% | 87.8-88.3% | 98.4% | 1.6% | 90.4-90.5% | 46.5-48.3% | 8.3-8.6% |
| First pass, joined, both | 94.6-94.9% | 88.6-89.3% | 98.7% | 1.0-1.1% | 91.4-91.5% | 46.5-48.3% | 8.6-9.0% |

- A crop list from the first pass does no measurable harm to rare species, and is as good for
  crop or not as a labelled one. A weed list from the first pass is what costs them: on ACRE it
  lists only the dominant weeds, so MATCH, SINAR and LOLPE lose out.
- Names improve more than actions. Decisions sum probabilities over classes, so at the same crops
  removed, harmful removed gains at most about 2 points on CropAndWeed.
- A list raises the confidence. At `defer_below` 0.9 CropAndWeed's harmful removed goes from 63-65%
  to 77%, and plants under the 0.7 hand-off from 15% to 10%, mostly because the threshold now
  means less. So compare decisions at the same crops removed, and tune thresholds again before
  acting on skewed probabilities.
- A weed species the list misses (each left off in turn, mean per species): CropAndWeed 79-80%
  right without a skew and 77-79% with it, called a crop 3% and 2%. On ACRE 45-46% and 35-36%,
  since CHEAL, which it is confused with, stays on the list.
- ACRE's rows at up to 1-2% of crops removed move a lot and aren't shown: about 16% of its labelled
  beans lie off the crop rows and take most removals, and they look like mislabelled weeds. An
  expert should check them first.

The evidence is thin: saved probabilities exist only for CropAndWeed, ACRE and PhenoBench (which
shows nothing here, at 99.5%), ACRE has 2 sessions, val shares sites with train, and no real
expert list has been tried. Lists are made per session, not live: a robot in the field would need
running counts.

## Unseen sites

Every dataset's val split shares sites and days with its train split, so the usual evaluation
can't show what a model does at a new site, nor what more diverse training data buys.
`scripts/evaluate_unseen.py` (settings: `configs/evaluate_unseen.yaml`, code: `weedseg.unseen`)
has runs of different combinations score the same plants, so that a dataset new to one run is
known to another. As set up, those are the three five-dataset DINOv3 seeds, the two public7 seeds
and the two public7+ImageWeeds seeds, on LincolnBeet, VCD and ImageWeeds. It uses the classifier
alone, on labelled boxes, so the detector plays no part.

- A run always scores a dataset's val split, so that runs can be compared on the same plants, and
  another split only if it never trained on that dataset. VCD's and ImageWeeds's train splits are
  listed because their val splits are thin: VCD's has no beans from a known site other than ACRE's,
  and ImageWeeds's only 43 redroot pigweeds. `test` stays out, for the final evaluation.
- Plants of a species the run has no class for are scored too, to see what it does with them.
- Partly labelled plants, such as weeds without a species, are scored by the probability of the
  classes their label rules out, not by the top class alone: on a plant of an unknown species the
  output is often flat, and a top class at 0.05 says little.
- Plants are grouped by dataset, split, label and site. The site is found in the image's `group`
  with a regular expression per dataset. ImageWeeds doesn't give its sites, and VCD's Montoldre is
  where ACRE was recorded, so it isn't a new place for a run trained on ACRE.

It writes `runs/unseen/unseen_<date>_<time>.md` and `.csv`: one column per run (A, B, ...), and
for each group of plants which runs are `new` to that dataset and which have `seen` it, then:

| Row | |
|---|---|
| right, right genus | Labelled plants; "right" is empty when the run has no class for the species |
| top class ruled out, ruled-out classes reach 0.7 | Partly labelled plants, instead of the two above |
| top class at 0.7 or more | How sure the run is; 0.7 is the export's hand-off to the zero-shot model |
| other_plant | The share whose top class is `other_plant` |
| keep / remove / defer | Given that it is a plant, at the rule file's `defer_below` |
| median top probability, most predicted | What the run does with these plants |

It cuts only the crops it needs, to `runs/unseen/crops/` (skipped when up to date), and keeps each
run's probabilities in `runs/unseen/<run>/<split>_<checkpoint>.npz`, so a stopped session goes on
where it was and only the table is made again. Each file is written when that run's split is done,
so stopping inside one loses that split. The plants are 5,248 in val (LincolnBeet 3,663, VCD 1,307,
ImageWeeds 278) and 6,911 in train (VCD 5,654, ImageWeeds 1,257), and a laptop CPU scores about 3
per second, so this is a job for the training machine. Lower `batch_size` and `num_workers` on a
small machine. `scripts/evaluate_rows.py` reads the same files (see Crop rows and spacing).

Results: pending.

## Crop rows and spacing

`weedseg.rows` says where each plant of an image stands against the crop rows: how far off the
nearest row, how close to the next plant along it and, where a session's plants are evenly
spaced, how far from where that spacing expects a plant. It knows nothing of species: it takes
positions in mm and which plants count as evidence for the rows, and judges every plant from the
other evidence plants only. Images cover less than a metre, so a session supplies what one image
can't: the direction of the rows, the distance between them, the usual gap along a row, and the
pitch, only when the gaps really are periodic. It isn't part of the export or the platform.

`scripts/evaluate_rows.py` (settings: `configs/evaluate_rows.yaml`) measures what that adds to the
classifier, from saved predictions, in about 5 minutes without a GPU. It writes
`runs/rows/rows_<date>_<time>.md` and `.csv`, with `_where.csv`, `_veto.csv`, `_alone.csv` and
`_sessions.csv` (every session's rows). It looks at three sets of plants:

- Position alone: every labelled plant, with rows from the labelled crops and no model.
- Two-stage detections, with rows from the plants called crop (or, as a best case, the labelled
  ones). Without a position its crops removed and harmful removed equal those of `evaluate_skew.py`
  without a skew: both count the plants found, where `by_dataset.csv` also counts those missed.
- Labelled boxes of datasets a run never trained on, as if the detector were perfect: the test of
  a new site. It needs the probabilities of `scripts/evaluate_unseen.py`, and a run without them
  is listed at the end of the report and left out. **No results yet.**

A position's worth is learned as a table of log likelihood ratios per cell (how far off the row,
how close to the next plant, how far from an expected plant), from plants other than those judged:
other sessions of any dataset, other sessions of the same dataset (tuned to the crop and the
place), or other datasets (all a new field could have). It counts only in an image whose fullest
row has at least 3 other plants, since any two make a line. It joins the classifier in two ways:

- Plain: the odds of a crop x ratio ^ trust.
- Stack: a logistic regression on the classifier's logit and the ratio, learned on the same plants.
  The fairest version, since it can give the position no weight. Its control, `classifier, stack`,
  has no position at all, so it can only recalibrate. A stack is worth what it adds beyond that.

Weeds include those the source didn't name (PhenoBench's, LincolnBeet's), which
`evaluate_skew.py` leaves out. "Weeds caught" is the share under the P(crop) threshold that loses
at most that share of the crops: it depends neither on a rule file nor on where 0.5 falls.

Results on val (2026-11-04, the two public7 DINOv3 seeds; frozen is one run of an older
combination, whose PhenoBench weeds aren't scored). Position alone, rows from the other labelled
crops:

| Dataset, crop | Sessions with a pitch | Pitch | Crops / weeds on a row (50 mm) | Row AUC | Weeds on a row: spacing AUC | Phase AUC |
|---|---|---|---|---|---|---|
| PhenoBench, beet | 12 of 12 | 212 mm | 96% / 25% | 0.91 | 0.79 | 0.80 |
| LincolnBeet, beet | 2 of 3 | 148 mm | 84% / 35% | 0.81 | 0.56 | 0.72 |
| ACRE, bean | 0 of 2 | - | 84% / 11% | 0.90 | 0.64 | - |
| CropAndWeed, maize | 1 of 24 | 177 mm | 62% / 26% | 0.73 | 0.64 | - |
| CropAndWeed, beet | 1 of 12 | 322 mm | 78% / 40% | 0.75 | 0.82 | too few |

With the classifier, weeds caught at up to 1% and 5% of crops lost, rows from the plants called
crop, stacks:

| Dataset, crops lost | Classifier | Control, same dataset | Rows, same dataset | Rows and spacing, same dataset | Rows and spacing, other datasets |
|---|---|---|---|---|---|
| CropAndWeed, 1% | 98.1-98.2% | 98.1% | 98.1% | 98.0-98.2% | 98.1% |
| CropAndWeed, 5% | 99.5% | 99.4-99.5% | 99.5-99.6% | 99.6% | 99.5-99.6% |
| PhenoBench, 1% | 96.9-97.3% | 96.7-97.3% | 97.7-98.3% | 97.9-98.5% | 97.2-97.7% |
| PhenoBench, 5% | 99.6-99.7% | 99.5-99.6% | 98.9-99.3% | 99.3% | 99.5-99.6% |
| ACRE, 1% | 16.1-38.5% | 12.9-25.7% | 13.4-23.8% | 13.6-26.3% | 15.7-30.7% |
| ACRE, 5% | 84.7-86.4% | 73.7-78.6% | 71.3-77.1% | 71.0-76.7% | 69.6-70.1% |
| CropAndWeed, frozen, 1% | 76.3% | 75.8% | 75.5% | 75.5% | 66.7% |
| CropAndWeed, frozen, 5% | 90.4% | 90.3% | 90.7% | 90.5% | 87.9% |

- A position says a lot alone and little the classifier doesn't. In images with a real row, it
  could put right 21 errors on CropAndWeed against 407-410 right plants it puts at risk (right
  crops off a row, right weeds on it), on PhenoBench 69-72 against 613-622, on ACRE 68-71 against
  793-815. The frozen classifier has more to fix but no better odds: 89 against 357 on CropAndWeed.
- The one gain is on PhenoBench, tuned on the same dataset: with rows and spacing, weeds called
  crop fall from 2.4-3.0% to 1.4-1.7% with crops called weeds level at 1.1%, 33-37 errors fixed
  and 5-6 caused, where the control fixes 9-16 and causes 17-20. That is about 30 plants of 7,400,
  and the classifier alone still catches a little more at 5% of crops lost.
- What spacing adds is mostly less harm from the rows. A weed on the row looks like a crop to the
  row alone: at full trust, PhenoBench weeds called crop reach 10.0-10.1% with rows, 3.5-3.7% with
  rows and spacing. In a stack it adds about 0.2 points on top of rows.
- Spacing could only be judged on beet. Beans show no period (ACRE 0 of 2 sessions, VCD's 0 of 3),
  and CropAndWeed's sessions of about 10 images are too short to show one, though single images of
  maize can look very regular.
- Tables don't travel. Learned on other crops, spacing is taken at the wrong strength: at full
  trust, PhenoBench weeds called crop reach 18.0-21.9%. Odds x ratio at full trust does harm in
  general, since label smoothing keeps the classifier's probabilities far from 0 and 1: CropAndWeed
  crops called weeds go from 1.7-1.8% to 5.9-6.1%.
- Rows from the labelled crops do no better than rows from the plants called crop, so finding the
  rows isn't what limits this: PhenoBench 97.7-98.4% at 1% of crops lost against 97.9-98.5%.
- A stack moves decisions by recalibrating alone. On the frozen run's CropAndWeed plants, harmful
  removed at up to 2% of crops goes from 61.3% to 74.6-75.5% in every same-dataset stack, and to
  75.1% in the control. So that gain isn't the position's.
- As a brake (a plant on a real row isn't removed) it costs more than it saves. On CropAndWeed,
  crops removed go from 0.08% to 0.04%, about one plant, and harmful removed from 63.0-64.8% to
  58.4-59.6%, or 61.7-63.2% for well-spaced plants only.
- Plant size against the image's fullest row helps only on ACRE: at 1% of crops lost 31.0-31.7%
  against the control's 12.9-25.7%, and 63.8-66.7% when the row is made of labelled crops. It does
  nothing on CropAndWeed or PhenoBench.
- ACRE's numbers can't be taken at their word: most crops the classifier calls weeds are off the
  rows (133-141 of 174-188, in images with a real row), and those "beans" look mislabelled (see
  Species lists). If they are weeds, the classifier and the rows are both right about them and the score
  says otherwise. An expert should check them first.

The evidence is thin, and this test can't show some of what rows might do. Val shares sites with
train, where the classifier is at its best: a new site is the open question. There are two seeds
and no confidence intervals; ACRE has 2 sessions, and 1% of its crops is about 12 plants. Images
are judged alone, whereas overlapping images could carry a row's history along and show the same
plant more than once. No dataset labels weeds that look just like the crop (volunteer beet in
beet), which only a position can catch.

## Test-time augmentation

Marco's idea: when a plant gets a low score, show the classifier several views of it and combine
the outputs. `scripts/evaluate_tta.py` (settings: `configs/evaluate_tta.yaml`, code: `weedseg.tta`)
tests it on the two public7 seeds. The prediction pipeline and the export don't use it.

- The views are things training has seen: the 8 combinations of a flip and quarter turns, and a
  tighter and a looser window (x0.85, x1.15), 10 in all. Crops are always resized to the input size,
  so a change of size only means a different window around the plant. Free-angle rotations and
  colour changes aren't in the test: training never saw the first, and doubling the colour jitter in
  training changed nothing (see Training).
- Every view is predicted once for every plant, not only for those with a low score. Which views,
  how they are combined and which plants get them are then worked out from the saved files, so
  trying another gate or view set needs no GPU.
- Probabilities are averaged, or counted as votes in a row of their own. Decisions sum class
  probabilities and compare them with `defer_below`, which ten votes can only do in steps of 0.1.
- A gate uses the views only for plants whose plain view has a top probability under it. That
  re-rolls only the low scores, so some rise over a threshold with nothing new known. Compare the
  gated rows with the same views without a gate, and decisions at the same crops removed.

It scores two sets of plants: the labelled boxes of the crop cache (classifier alone, all seven
datasets, so OPPD too), and the two-stage evaluation's detections (real boxes and the plant
threshold; CropAndWeed, ACRE and PhenoBench). The second counts the plants found, as
`scripts/evaluate_skew.py` does, so its decision shares run higher than `by_dataset.csv`.

It writes `runs/tta/tta_<date>_<time>.md` and `.csv`. Each table has one row per variant: the plain
crop, each view set averaged, then the main set by vote and behind each gate. Ranges are the spread
across seeds, and the plain row's spread is the noise to compare with. "Wrong to right" and "right
to wrong" count top classes changed against the plain view, so a gain shows as more of the first
than the second. The report starts with checks: the plain view must give the two-stage evaluation's
top classes, and the accuracy in the run's `metrics.json`.

Each view's probabilities are kept in `runs/tta/<run>/val_last/crops/` and `detections/`, one file
per view, so a stopped session goes on where it was and a view added later costs only that view. To
remake the report elsewhere, copy `runs/tta/`: it needs those files, the two-stage evaluations and
the species data, but no checkpoint, crop cache or GPU. The 4090 needs no preparation beyond the
public7 session's outputs. Its log there shows 33-67 s for a pass over the val crops and about 72 s
for classifying all detections, so this should take about 45 minutes for both seeds on an otherwise
idle machine, with about 440 MB of probabilities. A laptop CPU scores about 3 plants per second,
too slow for the full test; `max_plants` gives a smoke test.

ACRE's harmful removed at up to 1-2% of crops is too thin to read small moves: 1% of its crops is
about 12 plants. In a dry run with the saved probabilities rounded a second time, its row at 2% went
from 19.9% to 20.7% with nothing else changed.

The chance of a gain looked small beforehand, on the saved public7 val predictions (classifier
alone, seed 0 against seed 1). Under a top probability of 0.7 lie 11.8% of the crops and 67% of the
errors, so a gate looks in the right place. But on 78.8% of seed 0's errors, seed 1 picks the same
wrong class, 85% at 0.5-0.7 and 99.8% at 0.85 or more, and averaging removes noise, not a shared
bias. Averaging the two seeds' probabilities, as a rough upper limit since separately trained
models usually differ more than views of one image, moved species right on the plants found from
96.24% to 96.25% and balanced accuracy from 82.7% to 83.1%, and left harmful removed at the same
crops removed level. Of 19,682 plants, it changed the top class on 140: 51 became right, 53 became
wrong. That is two models rather than ten views, on three datasets without OPPD, which has the most
low scores (21% under 0.7), so it doesn't settle the question.

Results: pending.

## Deployment package

`scripts/export_twostage.py` (settings: `configs/export_twostage.yaml`) packages the
detector and classifier of `configs/twostage.yaml` for someone who runs the model outside
this repo, e.g. on a demo website. It writes `exports/weed_twostage_<date>/`:

| File | |
|---|---|
| `detector.safetensors`, `classifier.safetensors` | The weights only: plain tensors, no pickled code |
| `model.json` | What `weed_model.py` needs (image size, anchors, crop margin, normalisation), the classes with their scientific and English names, the training datasets of both stages with their licences and the stages each trains (`trains`, from each stage's own train.json: the detector leaves out images labelled with boxes only), and the hand-off thresholds |
| `weed_model.py` | Self-contained inference, `WeedModel.predict(rgb_image)`: torch, torchvision, timm, safetensors, numpy and opencv, nothing from `weedseg` |
| `requirements.txt` | Those packages (timm 1.0.20 or newer, for DINOv3) |
| `DINOv3_LICENSE.md` | With a DINOv3 classifier: the licence of its weights, which must go with them (a copy of Meta's, `src/weedseg/deploy/DINOv3_LICENSE.md`). The README then says "Built with DINOv3" and sums up the terms. Each backbone's licence is in `BACKBONES` of `weedseg.deploy.export`; an unknown one stops the export |
| `splits.csv` | Every image of the detector's and the classifier's datasets: dataset, path inside the dataset's image folder, sha256 of the file, group, and the split of each stage (empty when the stage didn't use it; the detector doesn't use images labelled with boxes only). The sha256 finds an image under any name, e.g. in the NeutraWeed platform, which stores images by it |
| `README.md` | For the person deploying it: install, input, output, hand-off, the species list, validation numbers, training data, splits and licences |

The README is made from `src/weedseg/deploy/README_template.md`, with the species,
training data and validation numbers of the runs read from their evaluation folders (run
steps 6 and 7 first). `weed_model.py` repeats the preprocessing of `predict_twostage.py`
(scaling, crop geometry, padding, normalisation) bit for bit. `tests/test_deploy.py` checks
it against the training code, so a change there that the package misses makes the tests
fail.

Each plant comes out with its box, outline (polygons), plant score, species, top 5, the
probabilities of all classes (4 significant digits, so small ones aren't rounded to 0) and a
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
