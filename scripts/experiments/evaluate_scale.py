"""How many plants does the plant detector still find when a photo shows the ground at another scale
than its training images? Settings: configs/evaluate_scale.yaml. See weedseg.scale_evaluation for
what is measured, and docs/twostage.md (Ground resolution and the detector).

Runs the two-stage pipeline's detector (boxes only) on val images shown at several scales; a GPU
is much faster, a CPU works. The detections are kept, so an interrupted run resumes.
Writes <output_dir>/scale_<date>_<time>.md and .csv, and prints the tables.
"""

from datetime import datetime

from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.scale_evaluation import write_report

CONFIG = "configs/evaluate_scale.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    stem = resolve(config["output_dir"]) / f"scale_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(config, stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
