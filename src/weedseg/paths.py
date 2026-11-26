"""Project-root anchored paths, so results don't depend on the working directory."""

import importlib
import sys
from functools import cache
from pathlib import Path
from types import ModuleType


@cache
def project_root() -> Path:
    """Return the first parent of this file that contains `pyproject.toml`."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("No pyproject.toml found above " + str(Path(__file__).resolve()))


def resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root; absolute paths are returned unchanged."""
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def import_script(path: str | Path) -> ModuleType:
    """Import a script from another scripts/ subfolder (e.g. "scripts/train/train_maskrcnn.py")
    as a module named after its file.

    Its folder goes on `sys.path`, as if the script sat next to the caller, so processes
    spawned later (which copy `sys.path`) can import it too and pickle its functions.
    """
    path = resolve(path)
    folder = str(path.parent)
    if folder not in sys.path:
        sys.path.append(folder)
    return importlib.import_module(path.stem)
