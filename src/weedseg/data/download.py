"""Dataset downloads that survive slow, unstable connections (e.g. through a VPN).

`download_file` fetches a file in pieces over several connections at once (HTTP range
requests), each written straight to its place in `<file>.part`. Progress is saved to
`<file>.part.json` every few seconds, so a dropped connection, a stopped script or a crash
costs only the last seconds: running again continues where it stopped. A connection that
stalls (slower than `min_speed_kbs` for `stall_s`) is dropped and reopened, and failed
attempts are retried with growing waits. Servers without range requests get one plain stream,
which restarts from zero when it breaks. The finished file is checked against its size and,
when known, its MD5 or SHA-256 before it is renamed into place.

A `.part` file without its `.json` (e.g. a browser's unfinished download, renamed) is taken
as the start of the file and continued.

Requests keep the `python-requests` user agent: digital.csic.es's bot check (Anubis)
challenges browser user agents, not scripts.
"""

import hashlib
import json
import math
import os
import queue
import shutil
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm

READ_SIZE = 1 << 16  # bytes per read from a connection: small, so slow links show progress
HASH_BLOCK = 1 << 23  # bytes per read when checking a checksum
MAX_WAIT_S = 120  # longest wait between two attempts
SAVE_EVERY_S = 5  # how often the progress file is rewritten


@dataclass
class TransferSettings:
    """How to use the connection; see configs/download.yaml (`transfer`)."""

    connections: int = 6
    chunk_mb: float = 256
    timeout_s: float = 120
    min_speed_kbs: float = 10
    stall_s: float = 120
    max_retries: int | None = None  # consecutive failures of a connection; None = no limit
    proxy: str | None = None

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> "TransferSettings":
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown transfer settings: {', '.join(sorted(unknown))}")
        return cls(**section)


class DownloadError(RuntimeError):
    """A download that retrying won't fix (wrong URL, a web page instead of the file, a
    file that changed on the server, a checksum mismatch)."""


class Stalled(requests.RequestException):
    """A connection slower than the minimum speed: reopen it."""


def file_name(url: str) -> str:
    return Path(urlparse(url).path).name


def fetch_files(files: list[dict[str, Any]], raw_dir: Path, settings: TransferSettings,
                keep_archives: bool = False) -> None:
    """Download a dataset's files (a config `download.files` list) into `raw_dir`.

    Each entry has a `url` and optionally `name` (default: from the URL), `size` (bytes),
    `md5` or `sha256`, and `extract` (unpack a .zip or .tar into `raw_dir`; the archive is
    kept in `raw_dir/.archives/` until then, and afterwards with `keep_archives`). A file
    finished in an earlier run (its marker in `raw_dir/.downloaded/`) is skipped.
    """
    for entry in files:
        name = entry.get("name") or file_name(entry["url"])
        marker = raw_dir / ".downloaded" / Path(name).stem
        if marker.exists():
            print(f"{name}: already downloaded, skipping")
            continue
        extract = entry.get("extract", False)
        target = raw_dir / ".archives" / name if extract else raw_dir / name
        download_file(entry["url"], target, settings, size=entry.get("size"),
                      md5=entry.get("md5"), sha256=entry.get("sha256"),
                      extract_after=extract)
        if extract:
            extract_archive(target, raw_dir)
            if not keep_archives:
                target.unlink()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()


def extract_archive(archive: Path, dest: Path) -> None:
    """Unpack a .zip or .tar into `dest`. Unsafe member paths (absolute, `..`) are refused or
    neutralised by zipfile and tarfile's "data" filter; a damaged zip member fails its CRC."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            try:
                for member in tqdm(zf.infolist(), desc=f"extracting {archive.name}",
                                   unit="file"):
                    zf.extract(member, dest)
            except zipfile.BadZipFile as error:
                raise DownloadError(f"{archive} is damaged ({error}); delete it and run "
                                    "again to download it anew") from error
    else:
        print(f"extracting {archive.name} ...")
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")


def download_file(url: str, path: Path, settings: TransferSettings | None = None,
                  size: int | None = None, md5: str | None = None, sha256: str | None = None,
                  extract_after: bool = False) -> None:
    """Download `url` to `path` (see the module doc). A complete `path` is left alone.

    `size`, when given, must match what the server reports, so a file that changed on the
    server is noticed. `extract_after` only counts in the disk space check.
    """
    if path.exists():
        return
    settings = settings or TransferSettings()
    path.parent.mkdir(parents=True, exist_ok=True)
    session = _session(settings)
    try:
        total, ranges, validator = _with_retries(lambda: _probe(session, url, settings),
                                                 settings, path.name)
    except DownloadError as error:
        raise DownloadError(f"{error}. If it downloads in a browser, save it as {path} and "
                            "run again: it is then only unpacked") from error
    finally:
        session.close()
    if size is not None and total is not None and total != size:
        raise DownloadError(f"{path.name}: the server reports {total} bytes, the config says "
                            f"{size}. The file may have changed: check it, then fix the config")
    total = total if total is not None else size
    if total is not None:
        _check_space(path, total * (2 if extract_after else 1))
    partial = path.with_name(path.name + ".part")
    if ranges and total:
        _Segmented(url, partial, total, validator, settings).run()
    else:
        _download_stream(url, partial, total, settings)
    _verify(partial, total, md5, sha256)
    partial.replace(path)


def _session(settings: TransferSettings) -> requests.Session:
    session = requests.Session()
    if settings.proxy:
        session.proxies = {"http": settings.proxy, "https": settings.proxy}
    return session


def _probe(session: requests.Session, url: str,
           settings: TransferSettings) -> tuple[int | None, bool, str]:
    """The file's size, whether the server serves byte ranges, and its ETag or date."""
    with session.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                     timeout=settings.timeout_s) as response:
        _check_response(response, url)
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified", "")
        if response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            total = content_range.rsplit("/", 1)[-1]
            return (int(total) if total.isdigit() else None), True, validator
        length = response.headers.get("Content-Length")
        return (int(length) if length else None), False, validator


def _check_response(response: requests.Response, url: str) -> None:
    response.raise_for_status()
    if (response.headers.get("Content-Type", "").startswith("text/html")
            and not urlparse(url).path.endswith((".html", ".htm"))):
        raise DownloadError(f"{url} returned a web page instead of the file (a login, a bot "
                            "check or a moved file)")


def _fatal(error: Exception) -> bool:
    """Errors a retry won't fix: ours, and HTTP 4xx other than timeouts and rate limits."""
    if isinstance(error, DownloadError):
        return True
    response = getattr(error, "response", None)
    return (response is not None and 400 <= response.status_code < 500
            and response.status_code not in (408, 429))


def _wait(failures: int, settings: TransferSettings, what: str, error: Exception,
          stop: threading.Event | None = None) -> None:
    """Wait before retry number `failures` (cut short by `stop`), or raise when out of
    retries."""
    if settings.max_retries is not None and failures > settings.max_retries:
        raise error
    wait = min(2 ** failures, MAX_WAIT_S)
    limit = "" if settings.max_retries is None else f"/{settings.max_retries}"
    tqdm.write(f"{what}: {type(error).__name__}: {str(error)[:120]}; "
               f"retry {failures}{limit} in {wait} s")
    _sleep(wait, stop)


def _clock() -> float:
    return time.monotonic()


def _sleep(seconds: float, stop: threading.Event | None = None) -> None:
    if stop is None:
        time.sleep(seconds)
    else:
        stop.wait(seconds)


def _with_retries(action, settings: TransferSettings, what: str):
    failures = 0
    while True:
        try:
            return action()
        except Exception as error:  # sorted out below
            if _fatal(error) or not isinstance(error, requests.RequestException):
                raise
            failures += 1
            _wait(failures, settings, what, error)


def _check_space(path: Path, needed: int) -> None:
    done = path.with_name(path.name + ".part")
    needed -= done.stat().st_size if done.exists() else 0
    free = shutil.disk_usage(path.parent).free
    if free < needed:
        raise DownloadError(f"{path.name}: needs {needed / 1e9:.1f} GB more, only "
                            f"{free / 1e9:.1f} GB free on {path.anchor}")


class _Segmented:
    """One file fetched in pieces by several connections; see the module doc."""

    def __init__(self, url: str, partial: Path, total: int, validator: str,
                 settings: TransferSettings):
        self.url, self.partial, self.total, self.settings = url, partial, total, settings
        self.state_path = partial.with_name(partial.name + ".json")
        self.chunk = max(1, int(settings.chunk_mb * (1 << 20)))
        self.count = math.ceil(total / self.chunk)
        self.key = {"url": url, "size": total, "validator": validator, "chunk": self.chunk}
        self.done = self._load_state()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.errors: list[BaseException] = []
        self.saved_at = 0.0

    def _length(self, i: int) -> int:
        return min(self.chunk, self.total - i * self.chunk)

    def _load_state(self) -> list[int]:
        """Bytes done per piece: from the progress file when it matches this file and these
        settings, else from a `.part` without one (taken as the start of the file)."""
        if self.partial.exists() and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if {k: state.get(k) for k in self.key} == self.key:
                return state["done"]
            tqdm.write(f"{self.partial.name}: the file or the piece size changed, "
                       "starting over")
            self.partial.unlink()
        elif self.partial.exists():
            prefix = self.partial.stat().st_size
            if prefix <= self.total:
                tqdm.write(f"{self.partial.name}: continuing an unfinished download "
                           f"({prefix / 1e6:.0f} MB)")
                return [max(0, min(self._length(i), prefix - i * self.chunk))
                        for i in range(self.count)]
            self.partial.unlink()
        return [0] * self.count

    def _save_state(self) -> None:
        temporary = self.state_path.with_name(self.state_path.name + ".tmp")
        temporary.write_text(json.dumps({**self.key, "done": self.done}), encoding="utf-8")
        os.replace(temporary, self.state_path)
        self.saved_at = _clock()

    def run(self) -> None:
        if not self.partial.exists():
            self.partial.touch()
        self._save_state()
        todo: queue.Queue[int] = queue.Queue()
        for i in range(self.count):
            if self.done[i] < self._length(i):
                todo.put(i)
        connections = max(1, min(self.settings.connections, todo.qsize()))
        tqdm.write(f"{self.partial.name[:-5]}: {self.total / 1e9:.2f} GB in {self.count} "
                   f"pieces, {todo.qsize()} to fetch over {connections} connections")
        self.bar = tqdm(total=self.total, initial=sum(self.done), unit="B", unit_scale=True,
                        desc=self.partial.name[:-5], smoothing=0.05)
        # Daemon threads: a stopped script must not wait for a connection to time out.
        workers = [threading.Thread(target=self._worker, args=(todo,), daemon=True)
                   for _ in range(connections)]
        try:
            for worker in workers:
                worker.start()
            while any(worker.is_alive() for worker in workers):
                for worker in workers:
                    worker.join(0.5)
        finally:
            self.stop.set()
            with self.lock:
                self._save_state()
            self.bar.close()
        if self.errors:
            raise self.errors[0]
        if any(self.done[i] < self._length(i) for i in range(self.count)):
            raise DownloadError(f"{self.partial.name}: pieces missing after all connections "
                                "stopped; run again to fetch them")
        self.state_path.unlink()

    def _worker(self, todo: "queue.Queue[int]") -> None:
        session = _session(self.settings)
        try:
            while not self.stop.is_set():
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                self._fetch(session, i)
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            self.errors.append(error)
            self.stop.set()
        finally:
            session.close()

    def _fetch(self, session: requests.Session, i: int) -> None:
        start, length = i * self.chunk, self._length(i)
        failures = 0
        while self.done[i] < length and not self.stop.is_set():
            offset = start + self.done[i]
            try:
                self._stream(session, i, offset, start + length - 1)
                if self.stop.is_set():
                    return
                if self.done[i] < length:  # the server ended the answer early
                    raise requests.ConnectionError("connection closed early")
            except Exception as error:  # sorted out below
                if _fatal(error) or not isinstance(error, requests.RequestException):
                    raise
                failures = 0 if self.done[i] > offset - start else failures
                failures += 1
                _wait(failures, self.settings, f"{self.partial.name[:-5]} piece {i + 1}",
                      error, self.stop)

    def _stream(self, session: requests.Session, i: int, offset: int, end: int) -> None:
        settings = self.settings
        with session.get(self.url, headers={"Range": f"bytes={offset}-{end}"}, stream=True,
                         timeout=settings.timeout_s) as response:
            _check_response(response, self.url)
            got = response.headers.get("Content-Range", "")
            if response.status_code != 206 or not got.startswith(f"bytes {offset}-"):
                raise DownloadError(f"{self.url}: asked for bytes {offset}-{end}, got status "
                                    f"{response.status_code} {got!r}")
            # Unbuffered: what the progress file counts is already with the OS, so it
            # survives a crash of this script.
            with open(self.partial, "r+b", buffering=0) as f:
                f.seek(offset)
                window_start, window_bytes = _clock(), 0
                for block in response.iter_content(READ_SIZE):
                    if self.stop.is_set():
                        return
                    block = block[:end + 1 - offset]
                    f.write(block)
                    offset += len(block)
                    window_bytes += len(block)
                    now = _clock()
                    with self.lock:
                        self.done[i] += len(block)
                        self.bar.update(len(block))
                        if now - self.saved_at >= SAVE_EVERY_S:
                            self._save_state()
                    if offset > end:
                        return
                    if now - window_start >= settings.stall_s:
                        speed = window_bytes / (now - window_start) / 1000
                        if speed < settings.min_speed_kbs:
                            raise Stalled(f"{speed:.1f} kB/s over {settings.stall_s:g} s")
                        window_start, window_bytes = now, 0


def _download_stream(url: str, partial: Path, total: int | None,
                     settings: TransferSettings) -> None:
    """One plain stream, for servers without range requests: a break restarts it."""
    session = _session(settings)
    failures = 0
    try:
        while True:
            try:
                with session.get(url, stream=True, timeout=settings.timeout_s) as response:
                    _check_response(response, url)
                    with open(partial, "wb") as f, tqdm(total=total, unit="B",
                                                        unit_scale=True,
                                                        desc=partial.name[:-5]) as bar:
                        for block in response.iter_content(READ_SIZE):
                            f.write(block)
                            bar.update(len(block))
                if total is None or partial.stat().st_size == total:
                    return
                raise requests.ConnectionError("connection closed early")
            except Exception as error:  # sorted out below
                if _fatal(error) or not isinstance(error, requests.RequestException):
                    raise
                failures += 1
                _wait(failures, settings, partial.name[:-5], error)
    finally:
        session.close()


def _verify(partial: Path, total: int | None, md5: str | None, sha256: str | None) -> None:
    """Check size and checksum; a mismatch keeps the file as `.bad` and starts over next time."""
    problem = None
    size = partial.stat().st_size
    if total is not None and size != total:
        problem = f"has {size} bytes, expected {total}"
    elif md5 or sha256:
        digest = hashlib.md5() if md5 else hashlib.sha256()
        with open(partial, "rb") as f, tqdm(total=size, unit="B", unit_scale=True,
                                            desc=f"checking {partial.name[:-5]}") as bar:
            while block := f.read(HASH_BLOCK):
                digest.update(block)
                bar.update(len(block))
        expected = (md5 or sha256).lower()
        if digest.hexdigest() != expected:
            problem = f"{'MD5' if md5 else 'SHA-256'} {digest.hexdigest()}, expected {expected}"
    if problem:
        bad = partial.with_name(partial.name[:-5] + ".bad")
        partial.replace(bad)
        raise DownloadError(f"{partial.name[:-5]} {problem}; kept as {bad.name}, run again to "
                            "download it anew")
