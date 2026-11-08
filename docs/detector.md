# Plant detector (boxes)

This branch (`bounding_box_only`) trains one model: a **plant detector** that puts a box around
every plant in a top-down field image, with one class, `plant`. No species and no outlines.
It is an alternative detector for the NeutraWeed platform, for users who don't need
segmentation. On the platform a project picks its detector and its species classifier
separately, so this detector can run with the classifier of the two-stage model from `master`,
which names plants from a crop around each box.

Since it needs neither species nor outlines, it can learn from many more datasets than the
segmentation line: any dataset whose **every plant has a box** (see [Datasets](#datasets)).

The branch is a separate line: it never merges back into `master`, which keeps the
segmentation models, the species classifier and the two-stage pipeline.

## The steps

Every script runs without arguments; its settings are in the config next to it.

| Script | Config | What it does |
|---|---|---|
| `scripts/download_datasets.py` | `configs/download.yaml` | Downloads the datasets ([data-format.md](data-format.md#downloading)) |
| `scripts/convert_<dataset>.py`, `convert_weedai.py` | `configs/<dataset>.yaml` | Converts each dataset to our COCO format (boxes, EPPO codes, split by group) |
| `scripts/preview_annotations.py` | `configs/preview.yaml` | Draws a COCO file's boxes on its images: check that a dataset boxes every plant |
| `scripts/combine_datasets.py` | `configs/combine.yaml` | Merges the converted datasets into one class, `plant` |
| `scripts/train_detector.py` | `configs/detector.yaml` | Trains, evaluates on val every `eval_every` epochs, keeps the best checkpoint |
| `scripts/evaluate_detector.py` | `configs/detector.yaml` | Evaluates a checkpoint on val or test |
| `scripts/export_detector.py` | `configs/export_detector.yaml` | Writes the [deployment package](#deployment-package) |
| `scripts/demo_detector.py` | `configs/demo_detector.yaml` | Runs a package on any images and draws the boxes |
| `scripts/run_detector.py` | `configs/run_detector.yaml` | Combine, train, evaluate (and export) in a row, restarting a step after a crash |

## Running it

1. Download and convert the datasets listed in `configs/combine.yaml`, each with taxonomy
   `species` and `min_plant_size_mm: 20` (the defaults). `data/` is not shared with the
   `master` folder.
2. Put torchvision's COCO weights (~170 MB) at `model.pretrained_path`
   (`weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, gitignored); the URL is in the
   config. Then the training machine needs no internet access. With `pretrained_path: null`
   the first run downloads them to the torch cache instead.
3. Optional smoke test, on any machine: set `max_train_images: 16`, `max_val_images: 8`,
   `epochs: 2`, `evaluate.max_images: 4` and a throwaway `run_name`, run training and
   evaluation, then undo those changes. On a CPU this takes a few minutes.
4. Run `scripts/run_detector.py` (combine, train, evaluate), or the three scripts one by one.
   Follow training with `uv run tensorboard --logdir runs/detector`.
5. Evaluate on `test` only for the final numbers, so choices aren't tuned on it.
6. Export the package (`scripts/export_detector.py`) and upload it to the platform.

An interrupted training resumes when re-run, from the last checkpoint: `last.pt` is saved
after every epoch and every `train.checkpoint_every` steps (200, a few minutes), and the
epoch under way continues with the batches it hadn't seen yet. A new experiment needs a new
run name. `run_name` is a template: `{dataset}` and `{taxonomy}` come from the last two folders
of `data.dir` (e.g. `public5_plant_min20mm`); for runs that change anything else, add your own
text (e.g. `_lr0.01`). Resuming refuses a run whose `model` settings or `data.dir` differ.

## Datasets

A detector learns every unboxed area of an image as "not a plant", so a dataset is useful only
when every plant in its images has a box or lies in an ignore region. The species don't
matter (a plant of unknown species is a plant), nor do the outlines.

| Dataset | Every plant boxed? | In `public5` |
|---|---|---|
| [CropAndWeed](datasets/cropandweed.md) | Yes, mostly: plants in the masks without a box become ignore regions. Some plants are left unlabelled altogether (seen in the previews), so some "detections on no plant" are real plants | yes |
| [ACRE](datasets/acre.md) | Yes; unknown and untrusted plants are ignore regions | yes |
| [PhenoBench](datasets/phenobench.md) | Yes: every plant pixel belongs to an instance | yes |
| [LincolnBeet](datasets/lincolnbeet.md) | Beets and weeds are boxed; the paper doesn't say whether every plant is, but in the images we checked they were | yes |
| [OPPD](datasets/oppd.md) | Boxed plant by plant (not checked whether stray plants in a tray are too); trays under one rig, not fields | yes |
| [GROUNDBASED_WEED](datasets/groundbased_weed.md) | No: TOMATO_2's tomatoes and plants of uncertain species are unboxed | no |
| [VCD](datasets/vcd.md) | No: only the crops are boxed | no |
| [ImageWeeds](datasets/imageweeds.md) | No: only the potted weeds are boxed | no |

The three left out are still converted (their converters work), so they can join later if
they are trained differently (below). Negative images (no plant over the size limit) are kept
where they show field: CropAndWeed, ACRE, PhenoBench and LincolnBeet. OPPD's are dropped:
they are mostly the first days of the same trays.

### New datasets (survey of 2026-11-06)

A search for datasets whose every plant is boxed, species not needed, picked first:

- [CornWeed](datasets/cornweed.md) (Germany, maize, RealSense on a tractor hoe and on BoniRob)
  and [WeedMaize](datasets/weedmaize.md) (Spain, commercial maize, compact camera from 1.3 m):
  converters written 2026-11-08; to be checked with `preview_annotations.py` before they join
  `configs/combine.yaml`.
- **Tobacco Aerial Dataset** (Pakistan, drone at 4 m): rejected 2026-11-08 after looking at
  its labels. Campaigns 1-2 label the weeds per pixel as one continuous carpet (no single
  plants); campaigns 3-8 label only the tobacco, the weeds being part of the background.
- Still to look at: Sugar Beets 2016 (Bonn; PhenoBench's area), the weed stem dataset WSD
  (vehicle at 1 m; maize, soybean, mung bean), Sudars (Latvia), DeepSeedling (cotton, USA),
  WeedsGalore (drone, 2.5 mm/px).

**"Positives only" for partly boxed datasets** (VCD, ImageWeeds, GROUNDBASED_WEED, and from
the survey MH-Weed16, 3SeasonWeedDet10, CoFly-WeedDB, Veridis), not done yet: their images
would teach what plants look like, from their boxes, without any background examples, so
unboxed plants aren't learned as soil. A small change on top of the ignore-region handling;
the cost is that the detector learns nothing about what is not a plant on those sites.

## Outputs

Everything goes to `runs/detector/<run_name>/` (gitignored):

- `last.pt`: model, optimiser and schedule state, for resuming.
- `best.pt`: the model with the best val box AP, plus its categories and config.
- `metrics.jsonl`: one line per epoch with the mean losses and, when evaluated, val AP/AP50.
- `config.yaml`: the config of the latest start or resume. `tensorboard/`: the logs.
- `eval/<split>_<checkpoint>/`, from `evaluate_detector.py`:
  - `metrics.json`: COCO box metrics (AP, AP50, AP75, by size, recall).
  - `by_dataset.csv`: per source dataset and for all: images, labelled plants, AP, AP50,
    the share of plants found, missed plants, detections on no labelled plant. A dense
    dataset weighs most in the overall numbers, so compare runs per dataset.
  - `by_species.csv`: plants found per EPPO code of their label, with the datasets they come
    from: which plants the detector misses.
  - `predictions.json` (COCO results), `previews/` (ground truth left, predictions right),
    `evaluation.json` (the split file, seed and `evaluate` settings used).

A plant counts as **found** when a detection with score ≥ `evaluate.score_threshold` (0.5)
overlaps its box with IoU ≥ `evaluate.iou_threshold` (0.5); each detection matches one plant,
best score first. Detections on an ignore region count neither way.

## Choices

- **Model:** torchvision's Faster R-CNN ResNet-50 FPN v2, fine-tuned from its COCO weights.
  It is the Mask R-CNN of `master` without the mask head, so the training code carries over;
  torchvision is BSD-3-Clause, with none of Ultralytics' AGPL terms. On COCO its box AP is
  46.7, against 37.0 for v1 (the platform's placeholder detector r50).
- **Ignore regions** (see [data-format.md](data-format.md#ignore-regions)) get the training
  label -1. The RoI heads of torchvision already skip proposals matched to such a box; the RPN
  is replaced by `IgnoreAwareRPN`, which does the same for anchors. A plant in an ignore region
  is thus learned neither as a plant nor as background.
- **Minimum plant size** is set at conversion (`min_plant_size_mm`, 20): smaller plants become
  ignore regions, so training and evaluation see the same data. The project targets plants
  past the seedling stage; a size in mm keeps the threshold valid across resolutions.
- **Evaluation** uses pycocotools, with the ignore regions counting for every category
  (`IgnoreAwareCOCOeval`). AP is the COCO mean over IoU 0.5-0.95, at up to
  `detections_per_image` detections per image.
- **Resolution:** images are resized so the short side is 1088 px (long side at most 1920):
  CropAndWeed stays at its native resolution, since 10% of its plants are under 22 px. The
  anchors start at 16 px instead of torchvision's 32.
- **Augmentation:** horizontal and vertical flips (top-down images have no "up") and mild
  colour jitter.
- **Optimiser:** SGD with linear warm-up and cosine decay, mixed precision on CUDA. The
  learning rate follows torchvision's reference recipe, scaled to the batch size.

## Memory

Mask R-CNN trained at 1920×1088 with a batch of 4 and mixed precision on the 24 GB training
GPU; without the mask head this model needs less. If memory runs out, lower
`train.batch_size` (and `lr` in proportion).

## Deployment package

`scripts/export_detector.py` writes `exports/plant_detector_<date>/`:

| File | Content |
|---|---|
| `detector.safetensors` | The weights (plain tensors, no pickled code) |
| `model.json` | Architecture and input settings, `min_plant_score`, smallest plant size, training data and licences, versions |
| `plant_detector.py` | Self-contained inference (`PlantDetector`), needing only torch, torchvision, safetensors, numpy and OpenCV |
| `requirements.txt` | Its Python packages |
| `splits.csv` | Every image of the training data with its sha256, dataset, group and `detector_split` |
| `README.md` | Usage, input and output, the val numbers per dataset and per species, training data, licences |

The README's table `| Dataset | Val images | Labelled plants | Plants found | ...` and the
`detector_split` column of `splits.csv` follow the two-stage export, which the platform
already reads. The platform's worker, though, runs only the two-stage Mask R-CNN v2 and the
placeholder Faster R-CNN v1 today: registering this package needs a `fasterrcnn_resnet50_fpn_v2`
case in `worker/nw/detector.py` and in `register_detector.py` of the platform (not done yet).

`plant_detector.py` repeats the training code's preprocessing exactly; `tests/test_deploy.py`
checks that it builds the trained architecture and gives the same boxes.
