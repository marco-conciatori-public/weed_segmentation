"""Run the steps of the plant detector in order, e.g. overnight on the training machine.
Settings: configs/run_detector.yaml (the steps and config files); every step reads its own
config, nothing is overridden here.

Each step runs in a process of its own. When that process crashes (e.g. the GPU driver
resets, which kills it without a Python error), the step is started again after
`restart_wait_s`, up to `max_restarts` times in all: training resumes from its last checkpoint
(saved every few minutes), the other steps redo their work. A Python error (a wrong setting,
a missing file) stops the runner, since a restart would fail again; after fixing the cause,
run it again: a finished training is not repeated.

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
import export_detector
import train_detector

from weedseg.config import load_config
from weedseg.console import log_console
from weedseg.detector.run import resolve_run_dir
from weedseg.paths import resolve

CONFIG = "configs/run_detector.yaml"
# Exit code of a step that raised a CUDA error as a Python exception: worth a restart.
CUDA_ERROR_EXIT = 75


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
    problems = config_problems(combine, detector, export, detector_path)
    if problems:
        raise SystemExit("The configs don't fit together:\n- " + "\n- ".join(problems))

    run_dir = resolve_run_dir(detector)
    steps = {
        "combine_datasets": (combine_datasets.main, combine_path, None),
        "train_detector": (train_detector.main, detector_path, run_dir),
        "evaluate_detector": (evaluate_detector.main, detector_path, run_dir),
        "export_detector": (export_detector.main, export_path, None),
    }
    unknown = [step for step in config["steps"] if step not in steps]
    if unknown:
        raise SystemExit(f"Unknown steps {unknown}; known: {', '.join(steps)}")
    print("Steps:")
    for step in config["steps"]:
        _, path, step_run_dir = steps[step]
        print(f"  {step:18} {path}" + (f" -> {step_run_dir}" if step_run_dir else ""))

    times, restarts = [], 0
    for step in config["steps"]:
        function, path, _ = steps[step]
        print(f"\n===== {step} ({path}) =====", flush=True)
        start = time.monotonic()
        while (exitcode := _run_in_process(function, path, log_path)) != 0:
            took = _duration(time.monotonic() - start)
            if not should_restart(exitcode):
                raise SystemExit(f"\n===== {step} failed after {took}. Fix the cause and run "
                                 "again (a finished training is not repeated) =====")
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


def config_problems(combine: dict, detector: dict, export: dict,
                    detector_path: str) -> list[str]:
    """Settings of the step configs that don't fit together, so a mistake shows up before
    hours of training. The combined folder's last part (plant_min<size>mm) depends on the
    converted data, so only the folder above it is checked."""
    problems = []
    combined = resolve(combine["output_dir"]) / (
        combine["name"] or "+".join(Path(folder).parent.name for folder in combine["datasets"]))
    if resolve(detector["data"]["dir"]).parent != combined:
        problems.append(f"data.dir of {detector_path} ({detector['data']['dir']}) is not in "
                        f"the folder combine_datasets writes ({combined})")
    if resolve(export["detector"]) != resolve(detector_path):
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
