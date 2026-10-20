"""YAML config loading."""

from pathlib import Path
from typing import Any

import yaml

from weedseg.paths import resolve


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; a relative `path` is taken relative to the project root.

    A config with a `base` key (another config's path) holds only what differs from that
    one: it is loaded on top of it, sections merged key by key (lists and plain values are
    replaced). E.g. configs/detector.yaml is configs/maskrcnn.yaml with another `data.dir`.
    """
    with open(resolve(path), encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base = config.pop("base", None)
    return merge(load_config(base), config) if base else config


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`base` with `override` on top, dicts merged recursively."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged
