"""Project-root anchored paths, so results don't depend on the working directory."""

from functools import cache
from pathlib import Path


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
