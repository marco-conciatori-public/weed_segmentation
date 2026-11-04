"""Does the classifier do better on several views of a plant than on one? Settings:
configs/evaluate_tta.yaml. See weedseg.tta for what is measured, and docs/twostage.md (Test-time
augmentation).

Keeps each view's probabilities in <output_dir>/<run>/<split>_<checkpoint>/crops/ and detections/,
one file per view, so a stopped session goes on where it was. With those files alone, the report can
be made again on another machine, e.g. with other view sets or gates. Writes
<output_dir>/tta_<date>_<time>.md and .csv, and prints the tables.
"""

from datetime import datetime

from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.tta import write_report

CONFIG = "configs/evaluate_tta.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    stem = resolve(config["output_dir"]) / f"tta_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(config, stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
