"""Classifier runs side by side: their evaluations and the two-stage predictions made with them.
Settings: configs/compare_runs.yaml. See weedseg.compare for what is shown.

Writes <output_dir>/compare_<date>_<time>.md (which run is which, and the table) and .csv, and
prints the table. scripts/run_twostage.py does the same after classifier variants.
"""

from datetime import datetime

from weedseg.compare import run_dir, write_comparison
from weedseg.config import load_config
from weedseg.paths import resolve

CONFIG = "configs/compare_runs.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    runs = [run_dir(entry) for entry in config["runs"]]
    stem = resolve(config["output_dir"]) / f"compare_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_comparison(runs, config["split"], config["checkpoint"],
                           resolve(config["twostage_dir"]), stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
