"""What skewing the classifier with a crop list and a weed list does to two-stage evaluations,
from their saved predictions: no GPU and no retraining. Settings: configs/evaluate_skew.yaml.
See weedseg.skew_evaluation for what is measured, and docs/twostage.md (Species lists).

Writes <output_dir>/skew_<date>_<time>.md (the tables, per dataset) and .csv, _missed.csv and
_lists.csv (every number per run, and every session's lists), and prints the tables.
"""

from datetime import datetime

from weedseg.compare import run_dir
from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.skew_evaluation import write_report

CONFIG = "configs/evaluate_skew.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    runs = [run_dir(entry) for entry in config["runs"]]
    stem = resolve(config["output_dir"]) / f"skew_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(runs, config, stem))
    print(f"-> {stem}.md, {stem.name}.csv, {stem.name}_missed.csv, {stem.name}_lists.csv")


if __name__ == "__main__":
    main()
