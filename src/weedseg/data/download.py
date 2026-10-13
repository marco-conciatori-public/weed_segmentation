"""Download helpers shared by the dataset modules."""

import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


class _ProgressReader:
    """File-like wrapper that reports bytes read to a tqdm bar."""

    def __init__(self, raw, bar: tqdm):
        self._raw = raw
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        data = self._raw.read(size)
        self._bar.update(len(data))
        return data


def download_and_extract_tar(url: str, dest: Path, marker: Path | None = None) -> None:
    """Stream a .tar from `url` and extract it into `dest` without keeping the archive.

    Extraction uses the "data" filter, which rejects absolute paths, links outside `dest`
    and other unsafe members. When `marker` is given, it is written after a successful
    extraction so a later run can skip the download.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        response.raw.decode_content = True
        with (
            tqdm(total=total, unit="B", unit_scale=True, desc=url.rsplit("/", 1)[-1]) as bar,
            tarfile.open(fileobj=_ProgressReader(response.raw, bar), mode="r|") as archive,
        ):
            archive.extractall(dest, filter="data")
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
