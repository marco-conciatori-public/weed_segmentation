# Plant detector (boxes): ${name}

This model finds every plant in a top-down field image and draws a box around it. It does not
name species and does not outline plants: pair it with a species classifier that works from
a crop around each box (such as the classifier of the NeutraWeed two-stage model) when the
species matter. It was built in the NeutraWeed project for robotic precision weeding. See
[Licences](#licences) for what it may be used for.

Exported ${created} from the training run `${run}`, epoch ${epoch}.

## How it works

A Faster R-CNN (torchvision `fasterrcnn_resnet50_fpn_v2`, fine-tuned from torchvision's
COCO-pretrained weights) with a single class, `plant`. For each plant it gives a box and a
*plant score*: how sure it is that this is a plant.

## Files

| File | Content |
|---|---|
| `plant_detector.py` | All the code needed to run the model (`PlantDetector`) |
| `model.json` | Settings, including `min_plant_score` (you can edit it) |
| `detector.safetensors` | The weights |
| `requirements.txt` | Python packages |
| `splits.csv` | The training data's images and the split each was in (see [Splits](#splits)) |

## Install

You need Python 3.10 or newer. In a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
# PyTorch first, the build for your machine (a CUDA version, or CPU only); see
# https://pytorch.org/get-started/locally/, for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Tested with Python ${python}, torch ${torch} and torchvision ${torchvision}. The weights are
in this folder, so nothing is downloaded at run time.

## Usage

```python
import json
from plant_detector import PlantDetector

model = PlantDetector("path/to/this/folder")  # the GPU if there is one
plants = model.predict_file("field.jpg")
# or, for an image already in memory (RGB, uint8, height x width x 3):
# plants = model.predict(rgb_array)
print(json.dumps(plants, indent=1))
```

Load the model once and reuse it for every image. `PlantDetector(folder, device="cpu")`
forces the CPU. On CUDA the model runs in half precision, as it was evaluated; `amp=False`
switches that off.

## Input

- **An RGB photo taken from straight above** (nadir view) of a field. All training images
  were taken this way, so photos taken at an angle are outside what the model knows.
- **Scale matters.** The model has no input for the ground resolution. It learned plants at
  about 0.4 to 1 mm per pixel, and it resizes every image so its short side is ${min_size} px
  (and its long side at most ${max_size} px). So **the short side of the image should cover
  roughly 0.45 to 1.1 m of ground**. A photo covering much more, such as a 3 m wide shot,
  makes the plants too small for the model: cut it into tiles of about 1 m, predict each
  tile, and shift the boxes back by the tile's offset.
- **Plant size.** Plants smaller than ${min_plant_size_mm} mm (longest side) were not labelled
  in training, so the model is not meant for seedlings that small.

## Output

`predict` returns a list with one entry per plant, the most certain first:

| Key | Meaning |
|---|---|
| `box` | `[x, y, width, height]` in pixels of the input image |
| `plant_score` | 0 to 1: how sure the detector is that this is a plant |

Plants with a plant score under `min_plant_score` (${min_plant_score}) are left out.
`predict(image, min_plant_score=...)` changes that for one call.

## Accuracy

The numbers on the validation split, images held out from training, per source dataset. A
labelled plant counts as found when a detection with a plant score of at least 0.5 overlaps
its box by at least 50% (IoU). A detection on no labelled plant is not always a mistake: some
datasets leave plants unlabelled.

${results_table}

Plants found per label (the species, or a coarser label, the annotators gave each plant):

${species_table}

## Training data

${dataset_table}

### Splits

`splits.csv` lists every image of the training data, one per row, with the split it was in,
so results on photos the model never saw can be told apart:

| Column | Meaning |
|---|---|
| `dataset` | The source dataset (table above) |
| `image` | The image's path inside the dataset's image folder |
| `sha256` | The SHA-256 of the image file as we downloaded it, to find it under any name (empty when the file was missing at export) |
| `group` | Its session, field and day, or tray, as the dataset records them. A group's images are all in the same split |
| `detector_split` | `train`, `val` or `test` |

${split_table}

`val` images were not trained on, but helped choose the checkpoint and settings. `test`
images were used for none of that, so they give the fairest measure.

## Licences

${licence_text}

Code: torchvision (BSD-3-Clause); the detector was fine-tuned from torchvision's
COCO-pretrained Faster R-CNN weights.
