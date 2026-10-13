"""Download helpers shared by the dataset modules."""

import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

CHUNK_SIZE = 1 << 20


def download_file(url: str, path: Path, timeout: float = 60, max_retries: int = 20) -> None:
    """Download `url` to `path`, resuming after network errors.

    Bytes go to `<path>.part` and are appended across attempts with HTTP range requests,
    so a dropped connection costs only a retry, not the whole file. The file is renamed to
    `path` once complete. Gives up after `max_retries` consecutive attempts that receive
    nothing; any progress resets the count. A complete `path` is not downloaded again.
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    failures = 0
    expected_size = None
    with tqdm(unit="B", unit_scale=True, desc=path.name) as bar:
        while True:
            done = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                    if done and response.status_code == 416:  # nothing left to fetch
                        break
                    response.raise_for_status()
                    if done and response.status_code != 206:  # server ignored the range
                        done = 0
                    remaining = int(response.headers.get("Content-Length", 0))
                    expected_size = done + remaining if remaining else None
                    bar.reset(total=expected_size)
                    bar.update(done)
                    with open(partial, "ab" if done else "wb") as f:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            f.write(chunk)
                            bar.update(len(chunk))
                            failures = 0
                break
            except requests.RequestException as error:
                failures += 1
                if failures > max_retries:
                    raise
                wait = min(2 ** failures, 60)
                resume = (f", resuming at {partial.stat().st_size / 1e6:.0f} MB"
                          if partial.exists() else "")
                tqdm.write(f"{path.name}: {type(error).__name__}, retry {failures}/{max_retries} "
                           f"in {wait} s{resume}")
                time.sleep(wait)

    size = partial.stat().st_size
    if expected_size is not None and size != expected_size:
        raise OSError(f"{path.name}: got {size} bytes, expected {expected_size}")
    partial.replace(path)


def download_and_extract_tar(url: str, dest: Path, marker: Path | None = None,
                             timeout: float = 60, max_retries: int = 20,
                             keep_archive: bool = False) -> None:
    """Download a .tar (resumable, see `download_file`) and extract it into `dest`.

    The archive is kept under `dest/.archives/` until extraction succeeds. Extraction uses
    the "data" filter, which rejects absolute paths, links outside `dest` and other unsafe
    members. When `marker` is given, it is written after a successful extraction so a later
    run can skip the part.
    """
    archive = dest / ".archives" / url.rsplit("/", 1)[-1]
    download_file(url, archive, timeout, max_retries)
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    if not keep_archive:
        archive.unlink()
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
