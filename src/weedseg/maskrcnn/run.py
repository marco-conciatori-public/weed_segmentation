"""Run directory and data files of a run config (configs/maskrcnn.yaml, configs/classifier.yaml):
its `data.dir`, `run_name`, `output_dir` and `evaluate` checkpoint and split."""

from pathlib import Path
from typing import Any

from weedseg.paths import resolve


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


def checkpoint_path(config: dict[str, Any], checkpoint: str | None = None) -> Path:
    """`checkpoint` (default `evaluate.checkpoint`): best or last of the run, or a path to a
    .pt file."""
    checkpoint = checkpoint or config["evaluate"]["checkpoint"]
    if checkpoint in ("best", "last"):
        return resolve_run_dir(config) / f"{checkpoint}.pt"
    return resolve(checkpoint)


def eval_dir(config: dict[str, Any]) -> Path:
    """Where evaluate_maskrcnn.py writes: <run>/eval/<split>_<checkpoint name>."""
    return (resolve_run_dir(config) / "eval"
            / f"{config['evaluate']['split']}_{checkpoint_path(config).stem}")
