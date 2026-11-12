"""Run directory and data files of a run config (configs/detector.yaml): its `data.dir`,
`run_name`, `output_dir`, `evaluate` checkpoint and splits, and `evaluate_scale` split. Also
which errors of a step are crashes worth a restart (scripts/run_detector.py)."""

from pathlib import Path
from typing import Any

from weedseg.paths import resolve

# Words in the error message of a crash caused by the machine, not by the settings or the data:
# a CUDA error (e.g. the GPU driver resets); DataLoader worker processes killed ("DataLoader
# worker (pid(s) ...) exited unexpectedly"; an error in the dataset code instead reads "Caught
# ... in DataLoader worker process 0"), or memory running out in the step itself, when the PC
# runs short of memory (e.g. other programs on a shared PC). The last two are PyTorch's and
# Windows' wording for running out of memory.
CRASH_SIGNS = ("CUDA", "DataLoader worker (pid", "not enough memory", "paging file is too small")


def split_file(config: dict[str, Any], split: str) -> Path:
    """COCO file of a split (train, val, test) inside `data.dir`."""
    return resolve(config["data"]["dir"]) / f"{split}.json"


def run_name(config: dict[str, Any]) -> str:
    """`run_name` with its placeholders filled in: {dataset} and {taxonomy} are the last two
    folders of `data.dir`."""
    data_dir = Path(config["data"]["dir"])
    values = {"dataset": data_dir.parent.name, "taxonomy": data_dir.name}
    try:
        return config["run_name"].format(**values)
    except KeyError as error:
        raise ValueError(f"Unknown placeholder {error} in run_name; "
                         f"available: {', '.join('{' + k + '}' for k in values)}") from None


def resolve_run_dir(config: dict[str, Any]) -> Path:
    return resolve(config["output_dir"]) / run_name(config)


def shared_run_dirs(configs: dict[str, dict[str, Any]]) -> list[str]:
    """Problems: detector configs (by path) that would train into the same run folder (a
    variant needs a run_name of its own)."""
    by_dir = {}
    for path, settings in configs.items():
        by_dir.setdefault(resolve_run_dir(settings), []).append(path)
    return [f"{' and '.join(paths)} share the run folder {run_dir}; give each its own run_name"
            for run_dir, paths in by_dir.items() if len(paths) > 1]


def is_crash(error: BaseException) -> bool:
    """Whether a step's error is a crash worth a restart (it may pass once the machine
    recovers) rather than a mistake that a restart would only repeat."""
    return isinstance(error, MemoryError) or any(sign in str(error) for sign in CRASH_SIGNS)


def checkpoint_path(config: dict[str, Any], checkpoint: str | None = None) -> Path:
    """`checkpoint` (default `evaluate.checkpoint`): best or last of the run, or a path to a
    .pt file."""
    checkpoint = checkpoint or config["evaluate"]["checkpoint"]
    if checkpoint in ("best", "last"):
        return resolve_run_dir(config) / f"{checkpoint}.pt"
    return resolve(checkpoint)


def eval_dir(config: dict[str, Any], split: str | None = None) -> Path:
    """Where evaluate_detector.py writes the evaluation of `split` (default
    `evaluate.split`): <run>/eval/<split>_<checkpoint name>."""
    split = split or config["evaluate"]["split"]
    return resolve_run_dir(config) / "eval" / f"{split}_{checkpoint_path(config).stem}"


def evaluated_splits(config: dict[str, Any]) -> list[str]:
    """The splits evaluate_detector.py evaluates: `evaluate.split`, then
    `evaluate.extra_splits`."""
    extra = config["evaluate"].get("extra_splits") or []
    split = config["evaluate"]["split"]
    return [split, *(s for s in extra if s != split)]


def scale_dir(config: dict[str, Any]) -> Path:
    """Where evaluate_scale.py writes: <run>/scale/<split>_<checkpoint name>."""
    return (resolve_run_dir(config) / "scale"
            / f"{config['evaluate_scale']['split']}_{checkpoint_path(config).stem}")
