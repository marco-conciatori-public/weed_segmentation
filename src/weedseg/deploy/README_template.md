# Weed species model (two-stage): ${name}

This model finds every plant in a top-down field image, outlines it, and names its species.
It knows ${num_species} species; a plant of any other species should come out as `other_plant`.
It was built in the NeutraWeed project for robotic precision weeding. This package is for
non-commercial research and demonstration (see [Licences](#licences)).

Exported ${created} from these training runs:

- detector: `${detector_run}`, epoch ${detector_epoch}
- classifier: `${classifier_run}`, epoch ${classifier_epoch}

## How it works

1. **Detector**: a Mask R-CNN (torchvision `maskrcnn_resnet50_fpn_v2`) with a single class,
   `plant`. For each plant it gives a box, a mask, and a *plant score*: how sure it is that
   this is a plant.
2. **Classifier**: a DINOv2 ViT-B/14 with a linear head (timm model
   `${backbone}`). It looks at a square crop around each plant the detector found and
   gives the probability of each species, assuming it is a plant.

The two scores answer different questions, so they stay separate. A detection can be
certainly a plant but of an uncertain species, or the other way round.

## Files

| File | Content |
|---|---|
| `weed_model.py` | All the code needed to run the model (`WeedModel`) |
| `model.json` | Settings, the species list and the hand-off thresholds (you can edit these) |
| `detector.safetensors` | Detector weights |
| `classifier.safetensors` | Classifier weights |
| `requirements.txt` | Python packages |

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

Tested with Python ${python}, torch ${torch}, torchvision ${torchvision} and timm ${timm}.
The weights are in this folder, so nothing is downloaded at run time.

## Usage

```python
import json
from weed_model import WeedModel

model = WeedModel("path/to/this/folder")  # loads both models; the GPU if there is one
plants = model.predict_file("field.jpg")
# or, for an image already in memory (RGB, uint8, height x width x 3):
# plants = model.predict(rgb_array)
print(json.dumps(model.to_json(plants), indent=1))
```

Load the model once (about 3 s) and reuse it for every image.
`WeedModel(folder, device="cpu")` forces the CPU. On CUDA the model runs in half
precision, as it was evaluated; `amp=False` switches that off.

A GPU is strongly recommended. On the RTX 4090 used for training, the validation images
took about 0.2 s each, evaluation included. On a laptop CPU, an image took 8 to 27 s; the
slowest had 51 plants, since the classifier runs once per plant. If GPU memory is short,
lower `classifier.batch_size` in `model.json`.

## Input

- **An RGB photo taken from straight above** (nadir view) of a field. All training images
  were taken this way, so photos taken at an angle are outside what the model knows.
- **Scale matters.** The model has no input for the ground resolution. It learned plants at
  about 0.4 to 1 mm per pixel, and it resizes every image so its short side is ${min_size} px
  (and its long side at most ${max_size} px). So **the short side of the image should cover
  roughly 0.45 to 1.1 m of ground**. A photo covering much more, such as a 3 m wide shot,
  makes the plants too small for the model: cut it into tiles of about 1 m, predict each
  tile, and shift the boxes and polygons back by the tile's offset.
- **Plant size.** Plants smaller than ${min_plant_size_mm} mm (longest side) were not labelled
  in training, so the model is not meant for seedlings that small.

## Output

`predict` returns a list with one entry per plant, the most certain first:

| Key | Meaning |
|---|---|
| `box` | `[x, y, width, height]` in pixels of the input image |
| `polygons` | The plant's outline: a list of polygons, each a list of `[x, y]` points (a plant cut in two, e.g. by a leaf gap, has several) |
| `mask` | The plant's mask, a boolean array the size of the image. Not included by `to_json`; `predict(..., masks=False)` leaves it out |
| `plant_score` | 0 to 1: how sure the detector is that this is a plant |
| `species` | [EPPO code](https://gd.eppo.int) of the most probable species (see [Species](#species)), or `other_plant` |
| `species_name` | Its scientific name |
| `common_names` | Its English names, as EPPO lists them. EPPO marks none as preferred, so the first is not necessarily the usual one |
| `species_probability` | 0 to 1: the classifier's probability of that species, assuming it is a plant |
| `top5` | The five most probable species, each with its probability |
| `send_to_zero_shot` | `true` when the plant should go to the zero-shot model (next section) |
| `zero_shot_reasons` | Why: any of `other_plant`, `low_species_probability`, `low_plant_score` |

Plants with a plant score under `handoff.min_plant_score` (${min_plant_score}) are left out.
`predict(image, min_plant_score=...)` changes that for one call.

The classifier was trained with label smoothing, so its probabilities rarely go above about
0.9, even when it is right: 0.9 already means "very sure".

## Hand-off to the zero-shot model

A plant gets `send_to_zero_shot: true` when any of these holds:

- its most probable class is `other_plant`: a species outside the list below;
- its species probability is under ${zero_shot_below_species_probability}
  (`handoff.zero_shot_below_species_probability`);
- its plant score is under ${zero_shot_below_plant_score} (`handoff.zero_shot_below_plant_score`):
  it may not be a plant at all.

These thresholds live in `model.json`, so you can change them without retraining. The
species threshold trades off how many plants the model names itself against how often it is
right. On the validation crops (labelled plants, scored by the classifier alone), not
counting those it calls `other_plant`:

${handoff_table}

## Species

${num_species} species, plus `other_plant`. English names: the first three EPPO lists. The
training plants are counted per source dataset. The validation accuracy is the
classifier's alone, on crops of labelled plants.

${species_table}

`other_plant` was trained on rarer taxa, which had too few examples for a class of their
own:

${other_plant_taxa}.

A plant of a species the model never saw at all may get any label, often with a lower
probability.

Species seen only in OPPD were photographed in trays in a greenhouse, in a daily time
series of the same few trays. How well the model recognises them in real fields is not
known.

Tomato, purple nutsedge (*Cyperus rotundus*) and bristly foxtail (*Setaria verticillata*)
were seen only in GROUNDBASED_WEED: tomato fields at one site in Spain, photographed with
one camera. How well the model recognises them elsewhere is not known.

## Accuracy

These are the numbers on the validation split, images held out from training, per source
dataset.

Both stages together. A plant counts as found when a detection with a plant score of at
least 0.5 overlaps its mask by at least 50% (IoU):

${results_table}

The classifier alone, on crops of the labelled plants:

${classifier_table}

How to read these numbers:

- **CropAndWeed** leaves many plants unlabelled. Some detections counted as wrong are real
  plants, so its numbers understate the model.
- **ACRE** has dense, small seedlings, with motion blur in many images, which is hard for
  any model. Its validation fields are bean fields full of fat-hen (*Chenopodium album*).
- **PhenoBench** gives a species only for sugar beet, so only its sugar beets are scored
  here.
- **OPPD** and **GROUNDBASED_WEED** have boxes, not outlines, so only the classifier is
  scored on them.
${partial_section}
## Training data

${dataset_table}

## Licences

The model was trained on datasets with non-commercial licences (CropAndWeed, PhenoBench and
OPPD; see the table above). So **use it only for non-commercial research and
demonstration**, and cite the datasets (papers above) when you present results. Commercial
use would need a model retrained without those datasets.

Code and pretrained weights: torchvision (BSD-3-Clause), timm and the DINOv2 weights
(Apache-2.0).
