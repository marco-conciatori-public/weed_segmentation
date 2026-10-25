"""Run an exported two-stage model on any images, labelled or not. Settings:
configs/demo_twostage.yaml.

Writes per image <name>.jpg (the plants drawn on it) and <name>.json (the plants) to the
config's output_dir. For scores on a labelled split, use scripts/predict_twostage.py.
"""

from weedseg.config import load_config
from weedseg.deploy.inference import run_inference

CONFIG = "configs/demo_twostage.yaml"


def main(config_path: str = CONFIG) -> None:
    run_inference(load_config(config_path))


if __name__ == "__main__":
    main()
