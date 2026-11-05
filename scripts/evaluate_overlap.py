"""Does the classifier do better on several real frames of a plant than on one? Settings:
configs/evaluate_overlap.yaml. See weedseg.overlap for what is measured, and docs/twostage.md
(Overlapping frames).

CPU only: it reads the two-stage evaluations and scripts/evaluate_tta.py's kept views, and needs the
images to place the frames. Keeps the frames' registrations in <output_dir>/registrations/, so only
the first run is slow. Writes <output_dir>/overlap_<date>_<time>.md, .csv, _links.csv,
_intervals.csv and _label_conflicts.csv, and prints the tables.
"""

from datetime import datetime

from weedseg.config import load_config
from weedseg.paths import resolve

CONFIG = "configs/evaluate_overlap.yaml"


def main(config_path: str = CONFIG) -> None:
    # Here, not at the top: on Windows each worker runs this file's top level again, and this import
    # brings in torch, which takes a worker from about 0.3 GB of memory to about 1.2 GB.
    from weedseg.overlap import write_report

    config = load_config(config_path)
    stem = resolve(config["output_dir"]) / f"overlap_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_report(config, stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
