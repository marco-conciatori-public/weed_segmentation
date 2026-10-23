"""Package the two-stage model for deployment outside this repo. Settings:
configs/export_twostage.yaml.

Writes <output_dir>/<name>/: both stages' weights (safetensors), model.json, weed_model.py
(self-contained inference), requirements.txt and README.md, with the species list,
training data and validation numbers of the runs (see weedseg.deploy.export). Run the
two-stage evaluation first (scripts/predict_twostage.py, scripts/evaluate_classifier.py),
or the README leaves those numbers out.
"""

from weedseg.config import load_config
from weedseg.deploy.export import export_package

CONFIG = "configs/export_twostage.yaml"


def main(config_path: str = CONFIG) -> None:
    export_package(load_config(config_path))


if __name__ == "__main__":
    main()
