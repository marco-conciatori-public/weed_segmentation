"""Does a plant's real size help tell its species, once the classifier has had its say? Settings:
configs/evaluate_size.yaml. See weedseg.size_evaluation for what is measured, and docs/twostage.md
(Plant size).

Works on the plain view that scripts/experiments/evaluate_tta.py keeps, so it needs no GPU and no checkpoint.
Writes <output_dir>/size_<date>_<time>.md and .csv, and prints the tables.
"""

from datetime import datetime

from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.size_evaluation import write_report

CONFIG = "configs/evaluate_size.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    stem = resolve(config["output_dir"]) / f"size_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(config, stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
