"""How many plants does the plant detector still find when a photo shows the ground at another
scale than its training images? Settings: configs/detector.yaml (`evaluate_scale`, plus `data`,
`run_name`, `output_dir` and `evaluate.checkpoint` to find the run and the data). See
weedseg.scale_evaluation for what is measured, and docs/detector.md (Scale).

Runs the detector on a split's images shown at several scales; a GPU is much faster, a CPU
works. Writes <run>/scale/<split>_<checkpoint>/report.md and report.csv; the detections are
kept there too (detections.jsonl), so an interrupted run resumes. Run
scripts/evaluate_detector.py first: the report checks its numbers against that evaluation.
"""

from weedseg.config import load_config
from weedseg.scale_evaluation import write_report

CONFIG = "configs/detector.yaml"


def main(config_path: str = CONFIG) -> None:
    print(f"-> {write_report(load_config(config_path))}")


if __name__ == "__main__":
    main()
