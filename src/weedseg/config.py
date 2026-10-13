"""YAML config loading."""

from pathlib import Path
from typing import Any

import yaml

from weedseg.paths import resolve


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; a relative `path` is taken relative to the project root."""
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)
