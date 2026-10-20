# Two-stage pipeline: plant detector + species classifier

An alternative to the species Mask R-CNN ([maskrcnn.md](maskrcnn.md)), scored by the same code
so the two can be compared file by file.

1. **Detector:** the same Mask R-CNN, trained on one class, `plant`. It finds and outlines the
   plants, whatever their species.
2. **Classifier:** DINOv2 ViT-B/14 (timm; Apache-2.0 code and weights) with a linear head,
   fine-tuned on square crops around each plant. Same classes as the species model,
   `other_plant` included.

Why: the species model's weak point is finding plants, not naming them. A one-class detector
doesn't split a plant's score over similar species (a plant at 0.4 / 0.3 / 0.2 over three
species can fall under the threshold), and it can later learn from datasets that label
plants but not species (PhenoBench, Sugar Beets, WeedsGalore). The classifier sees each
plant at full resolution and starts from a strong pretrained backbone.

## Steps

`scripts/run_twostage.py` runs steps 1-7 in order (settings: `configs/run_twostage.yaml`, the
list of steps), e.g. overnight on the training machine. Each step reads its own config;
the runner only orders them, after checking that the configs use the same data (see below).
Re-running is safe: crops already cut are skipped and trainings resume from `last.pt`
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
`twostage.yaml` names the detector and classifier configs and takes `best` (or `last`) of
their runs, so the prediction uses what was just trained. The runner checks that
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
CropAndWeed + ACRE that gives back 474 train plants. See `weedseg.data.plant`.

## Crops

A plant's crop is a square centred on its box, with `crops.margin` (25%) of the box's longer
side as context on each side, resized to `crops.input_size` (224 px). Crops at image edges are
padded with the ImageNet mean grey.

**Cache.** `prepare_crops.py` cuts every labelled plant once, 1.5 times larger than the crop
(`cache_context`, room for box jitter), with its mask:
`<data.dir>/crops_margin<m>_size<s>_context<c>/<split>/<annotation id>.jpg|.png` plus
`index_<split>.csv` (class, EPPO code, size in mm, dataset). Changing margin, size or context
makes a new cache folder next to the old one.

**Training augmentation:** box jitter (crop side ±20%, centre ±10% of the box), so a model
trained on labelled boxes copes with the detector's; flips and 90° rotations (top-down images
have no "up"); colour jitter.

**Background** (`crops.background`). Neighbouring plants can share a crop, e.g. a small weed
next to maize. `keep` (default) leaves the crop as it is; the classifier learns that the label
belongs to the plant in the centre. `dim` darkens the pixels outside the plant's mask to
`dim_factor` (30%): context stays visible, the target is obvious. `blank` greys them out, but
also loses leaf tips the mask misses (the detector's masks are rougher than the labels). With
`dim` or `blank`, training masks are grown or shrunk by up to `mask_jitter_px` to mimic the
detector. The cache always stores masks, so switching needs no new cache, only a new
classifier run.

## Training

AdamW with a low learning rate for the pretrained backbone (2e-5) and a high one for the new
head (1e-3), warm-up and cosine decay, label smoothing 0.1, stochastic depth 0.1, mixed
precision. `train.balance` can draw rare classes more often (`sqrt` or `inverse`); default
`none`. The best epoch by val accuracy is kept (`best.pt`); runs resume from `last.pt` like
the Mask R-CNN. Outputs: `runs/classifier/<run_name>/`.

`evaluate_classifier.py` scores the classifier on the labelled crops of a split, without the
detector: accuracy and balanced accuracy, per class, confusion matrix, `taxonomic_levels.csv`
(compare with the species model's, which also counts only plants it found), accuracy by plant
size (`accuracy_by_size.csv`, to see whether small plants suffer from their neighbours) and
`predictions.csv`.

## Prediction

`predict_twostage.py` runs the detector on the split, classifies every detection with a plant
score of at least `classifier.min_plant_score`, and writes the same outputs as
`evaluate_maskrcnn.py` to `runs/twostage/<detector run>__<classifier run>/eval/<split>/`.

Scores keep the species model's meaning: `class_scores[c]` = plant score × classifier
probability of `c`, and `score` is that of the most probable class. A low score can thus also
mean "maybe not a plant", and the decision rules (`defer_below`, summed per action) behave as
for the species model. Each detection also keeps its `plant_score`.

One difference in AP: the species model can return the same plant twice with two classes
(its post-processing makes one detection per proposal and class), which can add recall at
low scores; the two-stage pipeline returns each plant once, with its best class. The
confusion matrix, taxonomic levels and decisions, at score 0.5, are unaffected.
