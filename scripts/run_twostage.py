"""Run the steps of the two-stage pipeline in order. Settings: configs/run_twostage.yaml (the
steps and config files); every step reads its own config, nothing is overridden here.

Before the first step it checks that the configs fit together (same species data for the
plant copy, the classifier and the prediction; the detector on that plant copy), so a
mistake shows up before hours of training, not after. It stops at the first failing step;
re-running then skips or resumes the work already done (see the config).
"""

import faulthandler
import time

import evaluate_classifier
import evaluate_maskrcnn
import make_plant_dataset
import predict_twostage
import prepare_crops
import train_classifier
import train_maskrcnn

from weedseg.config import load_config
from weedseg.maskrcnn.run import checkpoint_path, resolve_run_dir
from weedseg.twostage import config_problems

CONFIG = "configs/run_twostage.yaml"


def main(config_path: str = CONFIG) -> None:
    # A crash inside CUDA can end the process before Python prints a traceback; this prints
    # at least the Python line it happened on.
    faulthandler.enable()
    config = load_config(config_path)
    twostage_path = config["twostage"]
    twostage = load_config(twostage_path)
    detector_path = twostage["detector"]["config"]
    classifier_path = twostage["classifier"]["config"]
    plant_path = config["plant_dataset"]
    detector, classifier = load_config(detector_path), load_config(classifier_path)

    problems = config_problems(load_config(plant_path), detector, classifier, twostage)
    if problems:
        raise SystemExit("The configs don't fit together:\n- " + "\n- ".join(problems))

    steps = {
        "make_plant_dataset": (make_plant_dataset.main, plant_path, None),
        "prepare_crops": (prepare_crops.main, classifier_path, None),
        "train_detector": (train_maskrcnn.main, detector_path, resolve_run_dir(detector)),
        "evaluate_detector": (evaluate_maskrcnn.main, detector_path, resolve_run_dir(detector)),
        "train_classifier": (train_classifier.main, classifier_path, resolve_run_dir(classifier)),
        "evaluate_classifier": (evaluate_classifier.main, classifier_path,
                                resolve_run_dir(classifier)),
        "predict_twostage": (predict_twostage.main, twostage_path, None),
    }
    unknown = [step for step in config["steps"] if step not in steps]
    if unknown:
        raise SystemExit(f"Unknown steps {unknown}; known: {', '.join(steps)}")
    print("Steps:")
    for step in config["steps"]:
        _, path, run_dir = steps[step]
        print(f"  {step:20} {path}" + (f" -> {run_dir}" if run_dir else ""))
    for name in ("detector", "classifier"):
        if twostage[name]["checkpoint"] not in ("best", "last"):
            print(f"Note: predict_twostage uses the fixed {name} checkpoint "
                  f"{twostage[name]['checkpoint']}, not the one trained here")
    if "predict_twostage" in config["steps"]:
        print(f"  prediction uses {checkpoint_path(detector, twostage['detector']['checkpoint'])}"
              f"\n                and "
              f"{checkpoint_path(classifier, twostage['classifier']['checkpoint'])}")

    times = []
    for step in config["steps"]:
        function, path, _ = steps[step]
        print(f"\n===== {step} ({path}) =====", flush=True)
        start = time.monotonic()
        try:
            function(path)
        except BaseException:
            print(f"\n===== {step} failed after {_duration(time.monotonic() - start)}; "
                  "fix the cause and run again (finished work is skipped or resumed) =====")
            raise
        times.append((step, time.monotonic() - start))
    print("\nDone: " + ", ".join(f"{step} {_duration(seconds)}" for step, seconds in times))


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


if __name__ == "__main__":
    main()
