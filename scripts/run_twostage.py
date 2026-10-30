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

Classifier variants (`classifier_variants`: configs holding only what differs from the
classifier's, each with its own run_name) are trained after the steps: each repeats the
classifier steps of the list (prepare_crops, train_classifier, evaluate_classifier,
predict_twostage, those listed) with its config, paired with the same detector. A variant may
train on other species data (its own data.dir, e.g. another combination): its prediction then
runs on that data's split. A variant that fails with a Python error is reported and the next
one starts. At the end all classifier runs are set side by side in
<log_dir>/compare_<date>_<time>.md and .csv (weedseg.compare).

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
from functools import partial
from pathlib import Path

import evaluate_classifier
import evaluate_maskrcnn
import make_plant_dataset
import predict_twostage
import prepare_crops
import train_classifier
import train_maskrcnn

from weedseg.compare import write_comparison
from weedseg.config import load_config
from weedseg.console import log_console
from weedseg.maskrcnn.run import checkpoint_path, resolve_run_dir
from weedseg.paths import resolve
from weedseg.twostage import config_problems, missing_data, other_data, shared_run_dirs

CONFIG = "configs/run_twostage.yaml"
# Exit code of a step that raised a CUDA error as a Python exception: worth a restart.
CUDA_ERROR_EXIT = 75
# The steps a classifier variant repeats, with its own config.
VARIANT_STEPS = ("prepare_crops", "train_classifier", "evaluate_classifier", "predict_twostage")


def main(config_path: str = CONFIG) -> None:
    # A crash inside CUDA can end the process before Python prints a traceback; this prints
    # at least the Python line it happened on.
    faulthandler.enable()
    config = load_config(config_path)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    log_path = resolve(config["log_dir"]) / f"run_twostage_{stamp}.log"
    log_console(log_path)
    print(f"Log: {log_path}")
    twostage_path = config["twostage"]
    twostage = load_config(twostage_path)
    detector_path = twostage["detector"]["config"]
    classifier_path = twostage["classifier"]["config"]
    plant_path = config["plant_dataset"]
    plant, detector = load_config(plant_path), load_config(detector_path)
    classifier = load_config(classifier_path)
    variants = {path: load_config(path) for path in config["classifier_variants"] or []}
    variant_steps = [step for step in config["steps"] if step in VARIANT_STEPS]

    problems = config_problems(plant, detector, classifier, twostage)
    for path, variant in variants.items():
        if not other_data(classifier, variant):  # else its prediction runs on its own data
            problems += [f"{path}: {problem}" for problem
                         in config_problems(plant, detector, variant, twostage)
                         if problem not in problems]
    problems += shared_run_dirs({classifier_path: classifier, **variants})
    if any(step in VARIANT_STEPS for step in config["steps"]):
        problems += missing_data({classifier_path: classifier, **variants})
    if (variants and "predict_twostage" in variant_steps
            and twostage["classifier"]["checkpoint"] not in ("best", "last")):
        problems.append("classifier variants need classifier.checkpoint best or last in "
                        f"{twostage_path}, or every prediction would use the same classifier")
    if problems:
        raise SystemExit("The configs don't fit together:\n- " + "\n- ".join(problems))

    def classifier_steps(path: str, settings: dict, variant: bool) -> dict:
        run_dir = resolve_run_dir(settings)
        predict = (partial(predict_twostage.main, classifier_config=path,
                           data_dir=settings["data"]["dir"]) if variant
                   else predict_twostage.main)
        return {"prepare_crops": (prepare_crops.main, path, None),
                "train_classifier": (train_classifier.main, path, run_dir),
                "evaluate_classifier": (evaluate_classifier.main, path, run_dir),
                "predict_twostage": (predict, twostage_path, None)}

    steps = {
        "make_plant_dataset": (make_plant_dataset.main, plant_path, None),
        "train_detector": (train_maskrcnn.main, detector_path, resolve_run_dir(detector)),
        "evaluate_detector": (evaluate_maskrcnn.main, detector_path, resolve_run_dir(detector)),
        **classifier_steps(classifier_path, classifier, variant=False),
    }
    unknown = [step for step in config["steps"] if step not in steps]
    if unknown:
        raise SystemExit(f"Unknown steps {unknown}; known: {', '.join(steps)}")
    print("Steps:")
    for step in config["steps"]:
        _, path, run_dir = steps[step]
        print(f"  {step:20} {path}" + (f" -> {run_dir}" if run_dir else ""))
    for path, variant in variants.items():
        print(f"Then classifier variant {path} -> {resolve_run_dir(variant)}:\n"
              f"  {', '.join(variant_steps) or 'no steps'}"
              + (f"\n  on its own data, {variant['data']['dir']}"
                 if other_data(classifier, variant) else ""))
    for name in ("detector", "classifier"):
        if twostage[name]["checkpoint"] not in ("best", "last"):
            print(f"Note: predict_twostage uses the fixed {name} checkpoint "
                  f"{twostage[name]['checkpoint']}, not the one trained here")
    if "predict_twostage" in config["steps"]:
        print(f"  prediction uses {checkpoint_path(detector, twostage['detector']['checkpoint'])}"
              f"\n                and "
              f"{checkpoint_path(classifier, twostage['classifier']['checkpoint'])}")

    times, restarts, failed = [], 0, []

    def run(step: str, function: Callable[[str], None], path: str) -> bool:
        """Run a step, again after each crash; False when it failed with a Python error."""
        nonlocal restarts
        print(f"\n===== {step} ({path}) =====", flush=True)
        start = time.monotonic()
        while (exitcode := _run_in_process(function, path, log_path)) != 0:
            took = _duration(time.monotonic() - start)
            if not should_restart(exitcode):
                print(f"\n===== {step} failed after {took} =====", flush=True)
                return False
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
        return True

    for step in config["steps"]:
        function, path, _ = steps[step]
        if not run(step, function, path):
            raise SystemExit("Fix the cause and run again (finished work is skipped or "
                             "resumed)")
    for path, variant in variants.items():
        variant_step_functions = classifier_steps(path, variant, variant=True)
        for step in variant_steps:
            function, step_path, _ = variant_step_functions[step]
            if not run(f"{step} [{Path(path).stem}]", function, step_path):
                failed.append(path)
                print(f"===== {path}: the rest of this variant is skipped; on to the next "
                      "=====", flush=True)
                break
    print("\nDone: " + ", ".join(f"{step} {_duration(seconds)}" for step, seconds in times)
          + (f" ({restarts} restart{'s' * (restarts > 1)} after crashes)" if restarts else ""))
    if failed:
        print(f"Variants that failed (see the log above; fix and run again): "
              f"{', '.join(failed)}")
    if variants:
        runs = [resolve_run_dir(classifier), *map(resolve_run_dir, variants.values())]
        stem = log_path.with_name(f"compare_{stamp}")
        print("\n" + write_comparison(runs, classifier["evaluate"]["split"],
                                      checkpoint_path(classifier).stem,
                                      resolve(twostage["output_dir"]), stem,
                                      load_config(config["compare_runs"])["crops_removed"]))
        print(f"Comparison: {stem}.md")


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
