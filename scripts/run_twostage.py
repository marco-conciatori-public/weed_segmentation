"""Run the steps of the two-stage pipeline in order. Settings: configs/run_twostage.yaml (the
steps and config files); every step reads its own config, nothing is overridden here.

Before the first step it checks that the configs fit together (same species data for the
plant copy, the classifier and the prediction; the detector on that plant copy), so a
mistake shows up before hours of training, not after.

Each step runs in a process of its own. When that process crashes (e.g. the GPU driver
resets, which kills it without a Python error), the step is started again after
`restart_wait_s`, up to `max_restarts` times in all: trainings resume from their last
checkpoint (saved every few minutes), the other steps skip or redo their work. A Python error
(a wrong setting, a missing file) stops the runner, since a restart would fail again; after
fixing the cause, run it again: finished work is skipped or resumed.

Everything printed also goes to <log_dir>/run_twostage_<date>_<time>.log, with the Python
lines where a step crashed.
"""

import faulthandler
import multiprocessing
import sys
import time
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import evaluate_classifier
import evaluate_maskrcnn
import make_plant_dataset
import predict_twostage
import prepare_crops
import train_classifier
import train_maskrcnn

from weedseg.config import load_config
from weedseg.console import log_console
from weedseg.maskrcnn.run import checkpoint_path, resolve_run_dir
from weedseg.paths import resolve
from weedseg.twostage import config_problems

CONFIG = "configs/run_twostage.yaml"
# Exit code of a step that raised a CUDA error as a Python exception: worth a restart.
CUDA_ERROR_EXIT = 75


def main(config_path: str = CONFIG) -> None:
    # A crash inside CUDA can end the process before Python prints a traceback; this prints
    # at least the Python line it happened on.
    faulthandler.enable()
    config = load_config(config_path)
    log_path = resolve(config["log_dir"]) / f"run_twostage_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_console(log_path)
    print(f"Log: {log_path}")
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

    times, restarts = [], 0
    for step in config["steps"]:
        function, path, _ = steps[step]
        print(f"\n===== {step} ({path}) =====", flush=True)
        start = time.monotonic()
        while (exitcode := _run_in_process(function, path, log_path)) != 0:
            took = _duration(time.monotonic() - start)
            if not should_restart(exitcode):
                raise SystemExit(f"\n===== {step} failed after {took}; fix the cause and run "
                                 "again (finished work is skipped or resumed) =====")
            if restarts >= config["max_restarts"]:
                raise SystemExit(f"\n===== {step} crashed (exit code {exitcode}) after {took}, "
                                 f"and the runner has already restarted {restarts} times; "
                                 "the machine looks unstable =====")
            restarts += 1
            print(f"\n===== {step} crashed (exit code {exitcode}) after {took}; restart "
                  f"{restarts} of {config['max_restarts']} in {config['restart_wait_s']} s "
                  "=====", flush=True)
            time.sleep(config["restart_wait_s"])
        times.append((step, time.monotonic() - start))
    print("\nDone: " + ", ".join(f"{step} {_duration(seconds)}" for step, seconds in times)
          + (f" ({restarts} restart{'s' * (restarts > 1)} after crashes)" if restarts else ""))


def should_restart(exitcode: int) -> bool:
    """Whether a step that ended with `exitcode` crashed (restart) rather than failed with a
    Python error (exit code 1: a restart would fail again)."""
    return exitcode not in (0, 1)


def _run_in_process(function: Callable[[str], None], path: str, log_path: Path) -> int:
    """Run function(path) in a new process, its output also appended to `log_path`; its exit
    code."""
    process = multiprocessing.get_context("spawn").Process(target=_step,
                                                           args=(function, path, log_path))
    process.start()
    try:
        process.join()
    except KeyboardInterrupt:  # Ctrl+C reaches the step too; don't leave it running
        process.terminate()
        raise
    return process.exitcode


def _step(function: Callable[[str], None], path: str, log_path: Path) -> None:
    """A step's process: function(path), with a CUDA error reported as worth a restart."""
    log = log_console(log_path)
    # The Python lines of a crash go to the log (the console still shows the crash itself).
    faulthandler.enable(file=log.file)
    try:
        function(path)
    except Exception as error:  # noqa: BLE001 (any error ends the step; printed here)
        traceback.print_exc()
        sys.exit(CUDA_ERROR_EXIT if "CUDA" in str(error) else 1)
    finally:
        log.close_line()


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


if __name__ == "__main__":
    main()
