"""Run the steps of the plant detector in order, e.g. overnight on the training machine.
Settings: configs/run_detector.yaml (the steps and config files); every step reads its own
config, nothing is overridden here.

Each step runs in a process of its own. When that process crashes (e.g. the GPU driver
resets, which kills it without a Python error) or ends with an error the machine caused (a
CUDA error, DataLoader workers killed, memory running out: weedseg.detector.run.is_crash), the
step is started again after `restart_wait_s`, up to `max_restarts` times in all: training
resumes from its last checkpoint (saved every few minutes), the scale test from its kept
detections, the other steps redo their work. Any other Python error (a wrong setting, a missing
file) stops the runner, since a restart would fail again; after fixing the cause, run it again:
a finished training is not repeated.

Detector variants (`detector_variants`: configs holding only what differs from the detector's,
each with its own run_name) are trained after the steps: each repeats the detector steps of
the list (train_detector, evaluate_detector, evaluate_scale, those listed) with its config. A
variant that fails with a Python error is reported and the next one starts. At the end all
runs are set side by side in <log_dir>/compare_<date>_<time>.md (weedseg.detector.compare).

Everything printed also goes to <log_dir>/run_detector_<date>_<time>.log, with the Python
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

import combine_datasets
import evaluate_detector
import evaluate_scale
import export_detector
import train_detector

from weedseg.config import load_config
from weedseg.console import log_console
from weedseg.detector.compare import write_comparison
from weedseg.detector.run import is_crash, resolve_run_dir, shared_run_dirs
from weedseg.paths import resolve

CONFIG = "configs/run_detector.yaml"
# Exit code of a step whose Python error is a crash (is_crash): worth a restart.
CRASH_EXIT = 75
# The steps a detector variant repeats, with its own config.
VARIANT_STEPS = ("train_detector", "evaluate_detector", "evaluate_scale")


def main(config_path: str = CONFIG) -> None:
    # A crash inside CUDA can end the process before Python prints a traceback; this prints
    # at least the Python line it happened on.
    faulthandler.enable()
    config = load_config(config_path)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    log_path = resolve(config["log_dir"]) / f"run_detector_{stamp}.log"
    log_console(log_path)
    print(f"Log: {log_path}")

    combine_path, detector_path = config["combine"], config["detector"]
    export_path = config["export"]
    combine, detector = load_config(combine_path), load_config(detector_path)
    export = load_config(export_path)
    variants = {path: load_config(path) for path in config.get("detector_variants") or []}
    problems = config_problems(combine, detector, export, detector_path)
    for path, variant in variants.items():
        problems += config_problems(combine, variant, None, path)
    problems += shared_run_dirs({detector_path: detector, **variants})
    if problems:
        raise SystemExit("The configs don't fit together:\n- " + "\n- ".join(problems))

    def detector_steps(path: str, settings: dict) -> dict:
        run_dir = resolve_run_dir(settings)
        return {"train_detector": (train_detector.main, path, run_dir),
                "evaluate_detector": (evaluate_detector.main, path, run_dir),
                "evaluate_scale": (evaluate_scale.main, path, run_dir)}

    steps = {
        "combine_datasets": (combine_datasets.main, combine_path, None),
        **detector_steps(detector_path, detector),
        "export_detector": (export_detector.main, export_path, None),
    }
    unknown = [step for step in config["steps"] if step not in steps]
    if unknown:
        raise SystemExit(f"Unknown steps {unknown}; known: {', '.join(steps)}")
    variant_steps = [step for step in config["steps"] if step in VARIANT_STEPS]
    print("Steps:")
    for step in config["steps"]:
        _, path, step_run_dir = steps[step]
        print(f"  {step:18} {path}" + (f" -> {step_run_dir}" if step_run_dir else ""))
    for path, variant in variants.items():
        print(f"Then detector variant {path} -> {resolve_run_dir(variant)}:\n"
              f"  {', '.join(variant_steps) or 'no steps'}")

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
            raise SystemExit("Fix the cause and run again (a finished training is not "
                             "repeated)")
    for path, variant in variants.items():
        variant_step_functions = detector_steps(path, variant)
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
        stem = log_path.with_name(f"compare_{stamp}")
        print("\n" + write_comparison([detector, *variants.values()], stem))
        print(f"Comparison: {stem}.md")


def config_problems(combine: dict, detector: dict, export: dict | None,
                    detector_path: str) -> list[str]:
    """Settings of the step configs that don't fit together, so a mistake shows up before
    hours of training. The combined folder's last part (plant_min<size>mm) depends on the
    converted data, so only the folder above it is checked. export None: not checked (a
    variant is not exported)."""
    problems = []
    combined = resolve(combine["output_dir"]) / (
        combine["name"] or "+".join(Path(folder).parent.name for folder in combine["datasets"]))
    if resolve(detector["data"]["dir"]).parent != combined:
        problems.append(f"data.dir of {detector_path} ({detector['data']['dir']}) is not in "
                        f"the folder combine_datasets writes ({combined})")
    if export is not None and resolve(export["detector"]) != resolve(detector_path):
        problems.append(f"the export's detector config is {export['detector']}, "
                        f"not {detector_path}")
    return problems


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
    """A step's process: function(path), with a crash (is_crash) reported as worth a
    restart."""
    log = log_console(log_path)
    # The Python lines of a crash go to the log (the console still shows the crash itself).
    faulthandler.enable(file=log.file)
    try:
        function(path)
    except Exception as error:  # noqa: BLE001 (any error ends the step; printed here)
        traceback.print_exc()
        sys.exit(CRASH_EXIT if is_crash(error) else 1)
    finally:
        log.close_line()


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


if __name__ == "__main__":
    main()
