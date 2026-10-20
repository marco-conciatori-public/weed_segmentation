# Mask R-CNN baseline

The first model for the prototype: torchvision's Mask R-CNN (ResNet-50 FPN v2), fine-tuned from
its COCO weights. It's Apache-2.0, like torchvision, so it carries none of Ultralytics' AGPL
terms. The goal is a working baseline, not the best model.

All settings live in `configs/maskrcnn.yaml`; both scripts run without arguments.

| Script | What it does |
|---|---|
| `scripts/train_maskrcnn.py` | Trains, evaluates on val every `eval_every` epochs, keeps the best checkpoint |
| `scripts/evaluate_maskrcnn.py` | Evaluates a checkpoint on val or test: metrics, confusion matrix, previews |
| `scripts/review_errors.py` | The worst errors of an evaluation, as images plus a CSV form for a plant expert |

## Running it

1. Convert each dataset with taxonomy `species` (`scripts/convert_cropandweed.py`,
   `scripts/convert_acre.py`; see `docs/datasets/`), combine
   them and choose the classes (`scripts/combine_datasets.py`, see
   [data-format.md](data-format.md)), and point `data.dir` at the combined folder, which
   holds `train.json`, `val.json` and `test.json`. A single converted folder works too.
2. Optional smoke test, on any machine: set `max_train_images: 16`, `max_val_images: 8`,
   `epochs: 2`, `evaluate.max_images: 4` and a throwaway `run_name`, run both scripts, then
   undo those changes. On a CPU this takes a few minutes.
3. Put torchvision's COCO weights (~180 MB) at `model.pretrained_path`
   (`weights/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth`, gitignored); the URL is in the
   config. This way the training machine needs no internet access. With `pretrained_path: null`
   the first run downloads them to the torch cache instead.
4. Run `scripts/train_maskrcnn.py`. Follow it with `uv run tensorboard --logdir runs/maskrcnn`.
5. Run `scripts/evaluate_maskrcnn.py` (`evaluate.split: val`). Use `test` only for the final
   numbers, so model choices aren't tuned on it.

An interrupted training resumes from the last finished epoch when re-run. A new experiment
needs a new run name. `run_name` is a template: `{dataset}` and `{taxonomy}` come from the
last two folders of `data.dir`; the converter names the latter after its settings (e.g.
`species_min50img_min20mm`). The default `{dataset}_{taxonomy}` thus changes by itself with
the data; for runs that change anything else, add your own text (e.g. `_lr0.01`). Resuming
refuses a run whose `model` settings or `data.dir` differ, but not other settings.

## Outputs

Everything goes to `runs/maskrcnn/<run_name>/` (gitignored):

- `last.pt`: model, optimiser and schedule state after the last finished epoch, for resuming.
- `best.pt`: the model with the best val mask AP, plus its categories and config.
- `metrics.jsonl`: one line per epoch with the mean losses and, when evaluated, val AP/AP50.
- `config.yaml`: the config of the latest start or resume. `tensorboard/`: the logs.
- `eval/<split>_<checkpoint>/`: `metrics.json`, `per_class.csv`, `confusion_matrix.csv/.png`,
  `predictions.json` (COCO results), `previews/` (ground truth left, predictions right) and,
  for species-level classes, `taxonomic_levels.csv` (see [Decisions](#decisions)).
  With `evaluate.rules`: `decisions.csv` (true role × action counts) and `decisions.json`
  (summary shares and the role of each class).

## Choices

- **Ignore regions** (see [data-format.md](data-format.md)) get the training label -1. The RoI
  heads of torchvision already skip proposals matched to such a box. The RPN is replaced by
  `IgnoreAwareRPN`, which does the same for anchors. A plant hidden in an ignore region is
  thus neither learned as a plant nor as background.
- **Minimum plant size** is set at conversion (`min_plant_size_mm` in the dataset config):
  plants whose box is shorter than this on its longer side, in mm, are written as ignore
  regions of the `ignore` category, so training and evaluation see the same data. The
  project targets plants past the seedling stage; a size in mm keeps the threshold valid
  across datasets of different resolution.
- **Evaluation** uses pycocotools, but the ignore regions count for every category
  (`IgnoreAwareCOCOeval`). Plain COCOeval would only apply each to its own category and
  count a correct "weed" on unlabelled vegetation as a false positive. AP is the COCO mean
  over IoU 0.5-0.95, at up to `detections_per_image` detections per image.
- **Resolution:** CropAndWeed is kept at its native 1920×1088, since 10% of its plants are
  under 22 px. The anchors start at 16 px instead of torchvision's 32.
- **Augmentation:** horizontal and vertical flips (top-down images have no "up") and mild
  colour jitter.
- **Optimiser:** SGD with linear warm-up and cosine decay, mixed precision on CUDA. The
  learning rate follows torchvision's reference recipe, scaled to the batch size.
- **Confusion matrix:** detections with score ≥ 0.5 matched to ground truth at IoU ≥ 0.5,
  whatever their class. The last row/column is "background": detections on no plant, and
  plants nothing detected. `evaluate.match_by` chooses mask or box IoU, for the confusion
  matrix, the decisions and the error review alike (AP is always COCO's mask and box AP).
  Mask IoU is strict with thin plants: a grass found in the right place but outlined
  roughly counts as a missed plant plus a detection on no plant. Box IoU counts it as found.

## Memory

At 1920×1088, a batch of 4 with mixed precision is expected to fit the 24 GB training GPU,
but this hasn't been measured yet. If memory runs out, lower `train.batch_size` (and `lr` in proportion), or `eval_batch_size`:
the predicted masks are pasted at full image size, up to `detections_per_image` per image.

## Decisions

The model recognises species; what the robot does with them is set by a rule file
(`evaluate.rules`, e.g. `configs/rules/example.yaml`), so the same model serves any field.
Each class gets a role from the most specific matching EPPO code (species, genus, family or
order) or the file's `default`: crop and neutral plants are kept, harmful ones removed, and
`other_plant` is deferred to a generic model. Crop status comes from the rules only: a
species marked as crop is always kept.

With the file's `defer_below`, the robot acts on its confidence in the action, not in the
species. The model gives each detection a probability per class (`class_scores` in
`predictions.json`, classes under 0.001 left out); the probabilities of all classes with the
same action are summed, and the plant is kept or removed when that sum reaches
`defer_below`, else deferred. A plant torn between two harmful grasses (0.5 and 0.4) is
removed at 0.9, where the top species alone (0.5) would defer it. Probability on
`other_plant` counts for no action, so a plant that may be an unknown species needs more
certainty. This covers decisions at genus or family level without naming the level: if all
of a family is harmful, its species' probabilities add up. Predictions from before
`class_scores` existed fall back to the top class's score; re-run the evaluation to use them.

The evaluation groups the confusion matrix by role: rows are the true role of each labelled
plant (from its ground-truth class), plus "no plant" for detections on no labelled plant;
columns are keep, remove, defer and missed. The summary gives the shares that matter for the
robot: crops removed, harmful weeds removed or left, neutral weeds removed, other_plant
deferred, and removals where no plant is labelled (possibly unlabelled plants). Confusions
inside one role (e.g. two harmful grasses) cost nothing here.

`taxonomic_levels.csv` looks at the plants the model found (matched in the confusion matrix)
and gives the share whose predicted class is right at each level: species, genus, family,
order and, with rules, role. It shows how wrong the wrong species are: if most errors keep
the genus or family, decisions taken at that level (rules on genus or family codes) lose
little. Missed plants, detections on no plant and pairs with `other_plant` are left out
(other_plant has no taxonomy); it uses the raw predictions, without `defer_below`.

## Error review for an expert

`scripts/review_errors.py` (settings: `review` in `configs/maskrcnn.yaml`) reads the
predictions of an evaluation, so it needs no GPU, and writes `<eval folder>/review/`. It is
replaced on each run. The errors are sorted into types by what they would cost the robot,
under the decision rules: crops that would be removed, harmful weeds kept or missed, neutral
plants removed, other_plant plants removed, detections where no plant is labelled, wrong
species with the same role, other wrong classes, other missed plants. Each type's folder
holds the `per_type` most confident errors (missed plants: the largest) as images, labels
left and model right with the plant framed in white, and `review.csv`: species by
scientific and English common name, confidence, role, robot action and size, with empty
`verdict` and `comment` columns for the expert. `README.txt` explains the images and the
verdicts; `summary.csv` counts every error of each type.

