"""Package the plant detector for deployment outside this repo. Settings:
configs/export_detector.yaml.

Writes <output_dir>/<name>/: the weights (safetensors), model.json, plant_detector.py
(self-contained inference), requirements.txt, splits.csv and README.md, with the training data
and validation numbers of the run (see weedseg.deploy.export). Run scripts/evaluate_detector.py
first, or the README leaves those numbers out.
"""

from weedseg.config import load_config
from weedseg.deploy.export import export_package

CONFIG = "configs/export_detector.yaml"


def main(config_path: str = CONFIG) -> None:
    export_package(load_config(config_path))


if __name__ == "__main__":
    main()
