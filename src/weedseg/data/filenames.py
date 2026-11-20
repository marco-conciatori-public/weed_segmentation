"""File names as they are spelled on disk.

Windows opens a file whatever the case of its name, Linux doesn't: a raw dataset whose
annotations say IMG_1.jpg for the file IMG_1.JPG converts and trains on Windows and fails on
Linux. These helpers find the spelling on disk, so reading raw data and the COCO files we
write work the same on both.
"""

import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath


@lru_cache(maxsize=256)
def _entries(folder: Path) -> tuple[frozenset[str], dict[str, tuple[str, ...]]] | None:
    """A folder's entry names, and the same grouped by casefolded name; None if not a folder."""
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    folded = defaultdict(list)
    for name in names:
        folded[name.casefold()].append(name)
    return frozenset(names), {key: tuple(value) for key, value in folded.items()}


def on_disk(relative: str | Path, root: Path) -> str | None:
    """`relative` (a path under `root`) as spelled on disk, "/"-separated: each part as it
    is if an entry has exactly that name, else the one entry whose name differs only in case.
    None if it doesn't exist. Raises ValueError if several entries differ from a part only in
    case (possible on Linux). `root` itself is taken as spelled right. Folder listings are
    cached, so call it after the files are in place."""
    folder, parts = root, []
    for part in PurePosixPath(Path(relative).as_posix()).parts:
        entries = _entries(folder)
        if entries is None:
            return None
        exact, folded = entries
        if part not in exact:
            matches = folded.get(part.casefold(), ())
            if not matches:
                return None
            if len(matches) > 1:
                raise ValueError(f"{folder}: several entries match {part!r} but for case: "
                                 f"{', '.join(sorted(matches))}")
            part = matches[0]
        parts.append(part)
        folder = folder / part
    return "/".join(parts)


def real_path(path: Path) -> Path:
    """A raw file to read, from a name we built: `path` if it exists, else the file whose name
    differs from it only in case, else `path` unchanged (opening it then fails as usual)."""
    if path.exists():  # on Windows whatever the case, on Linux only the exact name
        return path
    base = path.parent
    while not base.exists() and base != base.parent:
        base = base.parent
    found = on_disk(path.relative_to(base), base)
    return path if found is None else base / found
