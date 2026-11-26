"""Checks for converting several datasets in one go (scripts/data_prep/convert_datasets.py):
which ones have been downloaded, and whether their conversion settings agree.
"""

from pathlib import Path
from typing import Any

# Where download_datasets keeps the archives; a raw_dir holding only this has nothing unpacked.
ARCHIVE_DIR = ".archives"


def has_raw_data(raw_dir: Path) -> bool:
    """True when `raw_dir` holds anything besides the download archives."""
    return raw_dir.is_dir() and any(entry.name != ARCHIVE_DIR for entry in raw_dir.iterdir())


def setting_mismatches(sections: dict[str, dict[str, Any]], keys: list[str]) -> list[str]:
    """Describe each of `keys` whose value differs between the datasets' `convert` sections
    (dataset name -> section); a dataset without the key counts as `None`. Empty when all agree.
    """
    problems = []
    for key in keys:
        values: dict[str, list[str]] = {}
        for name, section in sections.items():
            values.setdefault(repr(section.get(key)), []).append(name)
        if len(values) > 1:
            problems.append(f"{key} differs: " + "; ".join(
                f"{value} in {', '.join(names)}" for value, names in values.items()))
    return problems
