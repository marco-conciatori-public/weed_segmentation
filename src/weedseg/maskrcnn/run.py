"""Run directory and data files of a configs/maskrcnn.yaml config."""

from pathlib import Path
from typing import Any

from weedseg.paths import resolve


def split_file(config: dict[str, Any], split: str) -> Path:
    """COCO file of a split (train, val, test) inside `data.dir`."""
    return resolve(config["data"]["dir"]) / f"{split}.json"


def run_name(config: dict[str, Any]) -> str:
    """`run_name` with its placeholders filled in: {dataset} and {taxonomy} are the last two
    folders of `data.dir`, {min_size} is e.g. `min20mm` (`all` without a minimum)."""
    data_dir = Path(config["data"]["dir"])
    min_size_mm = config["data"]["min_plant_size_mm"]
    values = {"dataset": data_dir.parent.name, "taxonomy": data_dir.name,
              "min_size": "all" if min_size_mm is None else f"min{min_size_mm:g}mm"}
    try:
        return config["run_name"].format(**values)
    except KeyError as error:
        raise ValueError(f"Unknown placeholder {error} in run_name; "
                         f"available: {', '.join('{' + k + '}' for k in values)}") from None


def resolve_run_dir(config: dict[str, Any]) -> Path:
    return resolve(config["output_dir"]) / run_name(config)
