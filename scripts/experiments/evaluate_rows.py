"""What a plant's place against the crop rows and the spacing along them adds to the classifier,
from saved predictions: no GPU and no retraining. Settings: configs/evaluate_rows.yaml. See
weedseg.row_evaluation for what is measured, and docs/twostage.md (Crop rows and spacing).

Writes <output_dir>/rows_<date>_<time>.md (the tables) and .csv, _where.csv, _veto.csv, _alone.csv
and _sessions.csv (every number per run, and every session's rows), and prints the tables.
"""

from datetime import datetime

from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.row_evaluation import write_report

CONFIG = "configs/evaluate_rows.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    stem = resolve(config["output_dir"]) / f"rows_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(config, stem))
    print(f"-> {stem}.md, {stem.name}.csv, {stem.name}_where.csv, {stem.name}_veto.csv, "
          f"{stem.name}_alone.csv, {stem.name}_sessions.csv")


if __name__ == "__main__":
    main()
