# Mask R-CNN baseline

The first model for the prototype: torchvision's Mask R-CNN (ResNet-50 FPN v2), fine-tuned from
its COCO weights. It's Apache-2.0, like torchvision, so it carries none of Ultralytics' AGPL
terms. The goal is a working baseline, not the best model.

All settings live in `configs/maskrcnn.yaml`; both scripts run without arguments.

| Script | What it does |
|---|---|
| `scripts/train_maskrcnn.py` | Trains, evaluates on val every `eval_every` epochs, keeps the best checkpoint |
| `scripts/evaluate_maskrcnn.py` | Evaluates a checkpoint on val or test: metrics, confusion matrix, previews |

## Running it

1. Convert the data for the chosen taxonomy (`scripts/convert_cropandweed.py`) and point
   `data.dir` at its folder, which holds `train.json`, `val.json` and `test.json`.
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
  `predictions.json` (COCO results) and `previews/` (ground truth left, predictions right).

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
- **Confusion matrix:** detections with score ≥ 0.5 matched to ground truth at mask IoU ≥ 0.5,
  whatever their class. The last row/column is "background": detections on no plant, and
  plants nothing detected.

## Memory

At 1920×1088, a batch of 4 with mixed precision is expected to fit the 24 GB training GPU,
but this hasn't been measured yet. If memory runs out, lower `train.batch_size` (and `lr` in proportion), or `eval_batch_size`:
the predicted masks are pasted at full image size, up to `detections_per_image` per image.
