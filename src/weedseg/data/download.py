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

`fetch_all` downloads the files of several datasets. A file that brings no new byte for
`give_up_after_min` minutes (the server is down, or can't serve part of it) is set aside, its
progress kept, so the next files get their turn. Set-aside files get another pass after the
others, as long as the previous pass brought anything new; a file that fails for a reason a
retry won't fix doesn't stop the others either. The unfinished files are listed at the end.

`fetch_share` copies a folder of a Nextcloud public share (a https://<server>/s/<token>
link) file by file, through the share's WebDAV address: the folder is listed, and the files
not there yet are fetched whole, several at once (one connection each), each continuing from
its `<file>.part` after a break. Meant for folders of many small files; the share's own
"download" button builds a zip on the fly, which can't be continued after a break.

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
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests
from tqdm import tqdm

READ_SIZE = 1 << 16  # bytes per read from a connection: small, so slow links show progress
HASH_BLOCK = 1 << 23  # bytes per read when checking a checksum
MAX_WAIT_S = 120  # longest wait between two attempts
SAVE_EVERY_S = 5  # how often the progress file is rewritten
DAV = "{DAV:}"  # WebDAV's XML namespace
SHARE_LISTING = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:resourcetype/>'
                 '<d:getcontentlength/></d:prop></d:propfind>')  # what a listing asks for


@dataclass
class TransferSettings:
    """How to use the connection; see configs/download.yaml (`transfer`)."""

    connections: int = 6
    chunk_mb: float = 256
    timeout_s: float = 120
    min_speed_kbs: float = 10
    stall_s: float = 120
    max_retries: int | None = None  # consecutive failures of a connection; None = no limit
    give_up_after_min: float | None = 30  # without a new byte: set the file aside; None = never
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


class NoProgress(RuntimeError):
    """A file that brought no new byte for `give_up_after_min`: set it aside for now.
    `gained` is what it brought before that, in bytes."""

    def __init__(self, message: str, gained: int = 0):
        super().__init__(message)
        self.gained = gained


@dataclass
class Unfinished:
    """A file left unfinished by a pass of `fetch_files`."""

    name: str
    target: Path
    reason: str
    retry: bool  # set aside for lack of progress: worth another pass


def file_name(url: str) -> str:
    return Path(urlparse(url).path).name


def fetch_all(datasets: list[tuple[str, dict[str, Any], Path]],
              settings: TransferSettings) -> None:
    """Download the files of several datasets: (name, config `download` section, raw_dir).

    Passes over all files (see `fetch_files`) are repeated for the files set aside for lack
    of progress, as long as the previous pass brought anything new. Raises DownloadError
    listing the unfinished files, if any.
    """
    failed: list[Unfinished] = []  # failed for a reason a retry won't fix
    while True:
        progress, set_aside = False, []
        for name, download, raw_dir in datasets:
            tqdm.write(f"===== {name} -> {raw_dir}")
            made, unfinished = fetch_files(download["files"], raw_dir, settings,
                                           download.get("keep_archives", False),
                                           skip={item.target for item in failed})
            progress |= made
            for item in unfinished:
                (set_aside if item.retry else failed).append(item)
        if not set_aside or not progress:
            break
        tqdm.write(f"===== {len(set_aside)} files set aside: another pass")
    if failed or set_aside:
        lines = [f"  {item.name}: {item.reason}" for item in failed + set_aside]
        raise DownloadError("Unfinished downloads:\n" + "\n".join(lines) + "\nProgress is "
                            "kept: run again later. A file that always stops at the same "
                            "point may be damaged on the server.")


def fetch_files(files: list[dict[str, Any]], raw_dir: Path, settings: TransferSettings,
                keep_archives: bool = False,
                skip: set[Path] | frozenset = frozenset()) -> tuple[bool, list[Unfinished]]:
    """One pass over a dataset's files (a config `download.files` list), into `raw_dir`.

    Each entry has a `url` and optionally `name` (default: from the URL), `size` (bytes),
    `md5` or `sha256`, and `extract` (unpack a .zip or .tar into `raw_dir`; the archive is
    kept in `raw_dir/.archives/` until then, and afterwards with `keep_archives`). An entry
    can instead be a folder of a Nextcloud public share: `share` (the link) and `path` (the
    folder), copied into `raw_dir/<path>` by `fetch_share`, optionally with `name` (default:
    the folder's) and `size` (all its files together). A file finished in an earlier run (its
    marker in `raw_dir/.downloaded/`) is skipped, and so are the download paths in `skip`. A
    file that fails, or brings no new byte for `give_up_after_min`, is left for later and the
    next one starts.

    Returns whether anything new came in, and the files left unfinished.
    """
    progress, unfinished = False, []
    for entry in files:
        share = entry.get("share")
        name = entry.get("name") or (PurePosixPath(entry["path"]).name if share
                                     else file_name(entry["url"]))
        marker = raw_dir / ".downloaded" / Path(name).stem
        if marker.exists():
            print(f"{name}: already downloaded, skipping")
            continue
        extract = not share and entry.get("extract", False)
        if share:
            target = raw_dir / entry["path"]
        else:
            target = raw_dir / ".archives" / name if extract else raw_dir / name
        if target in skip:
            continue
        try:
            if share:
                fetch_share(share, entry["path"], target, settings, size=entry.get("size"))
            else:
                download_file(entry["url"], target, settings, size=entry.get("size"),
                              md5=entry.get("md5"), sha256=entry.get("sha256"),
                              extract_after=extract)
            if extract:
                extract_archive(target, raw_dir)
                if not keep_archives:
                    target.unlink()
        except NoProgress as error:
            tqdm.write(f"{name}: {error}; set aside, it gets another try after the others")
            unfinished.append(Unfinished(name, target, str(error), retry=True))
            progress |= error.gained > 0
            continue
        except (DownloadError, requests.RequestException) as error:
            tqdm.write(f"{name}: {type(error).__name__}: {error}")
            unfinished.append(Unfinished(name, target, f"{type(error).__name__}: {error}",
                                         retry=False))
            continue
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        progress = True
    return progress, unfinished


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


def fetch_share(share: str, folder: str, dest: Path, settings: TransferSettings | None = None,
                size: int | None = None) -> None:
    """Copy `folder` of the Nextcloud public share at link `share` into `dest`, file by file
    (see the module doc). Files already in `dest` with the size the server lists are skipped.

    `size`, when given, must match the folder's total on the server, so a changed folder is
    noticed.
    """
    settings = settings or TransferSettings()
    root = _share_dav(share)
    session = _session(settings)
    try:
        files = _with_retries(lambda: _list_share(session, root, folder, settings), settings,
                              folder)
    finally:
        session.close()
    if not files:
        raise DownloadError(f"{share}: no files in {folder!r}")
    total = sum(file.size for file in files)
    if size is not None and total != size:
        raise DownloadError(f"{folder}: the server lists {total} bytes, the config says {size}. "
                            "The folder may have changed: check it, then fix the config")
    _ShareFolder(files, dest, PurePosixPath(folder).name, settings).run()


@dataclass
class _ShareFile:
    """A file in a share folder: its path in the folder, URL and size."""

    path: str
    url: str
    size: int


def _share_dav(share: str) -> str:
    """The WebDAV address of a Nextcloud public share link."""
    parts = urlparse(share)
    prefix, found, token = parts.path.rstrip("/").rpartition("/s/")
    if not found or not token or "/" in token:
        raise DownloadError(f"{share} is not a Nextcloud share link (https://<server>/s/<token>)")
    return f"{parts.scheme}://{parts.netloc}{prefix}/public.php/dav/files/{token}/"


def _list_share(session: requests.Session, root: str, folder: str,
                settings: TransferSettings) -> list[_ShareFile]:
    """The files in `folder` of the share at WebDAV address `root`, subfolders included."""
    start = root + quote(folder.strip("/")) + "/"
    prefix = unquote(urlparse(start).path)
    todo, found = [start], []
    while todo:
        url = todo.pop()
        # Nextcloud lists a public share only to requests marked as coming from its own web
        # page; others get 401 Unauthorized.
        response = session.request("PROPFIND", url, data=SHARE_LISTING,
                                   timeout=settings.timeout_s,
                                   headers={"Depth": "1", "Content-Type": "application/xml",
                                            "X-Requested-With": "XMLHttpRequest"})
        _check_response(response, url)
        try:
            entries = list(ElementTree.fromstring(response.content).iter(f"{DAV}response"))
        except ElementTree.ParseError as error:
            raise DownloadError(f"{url}: the answer is not a folder listing") from error
        for entry in entries:
            href = urljoin(url, entry.findtext(f"{DAV}href", ""))
            path = unquote(urlparse(href).path)
            if path.rstrip("/") == unquote(urlparse(url).path).rstrip("/"):
                continue  # the folder itself
            relative = PurePosixPath(path[len(prefix):])
            if not path.startswith(prefix) or ".." in relative.parts:
                raise DownloadError(f"{url}: unexpected entry {href}")
            if entry.find(f".//{DAV}collection") is not None:
                todo.append(href)
                continue
            length = entry.findtext(f".//{DAV}getcontentlength")
            if length is None:
                raise DownloadError(f"{href}: the server lists no size")
            found.append(_ShareFile(str(relative), href, int(length)))
    return sorted(found, key=lambda file: file.path)


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
                            "check, a moved file, or a link disabled for too much traffic, "
                            "as Dropbox does for a while)")


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


def _give_up_s(settings: TransferSettings) -> float | None:
    minutes = settings.give_up_after_min
    return None if minutes is None else minutes * 60


def _with_retries(action, settings: TransferSettings, what: str):
    failures, started = 0, _clock()
    while True:
        try:
            return action()
        except Exception as error:  # sorted out below
            if _fatal(error) or not isinstance(error, requests.RequestException):
                raise
            give_up_s = _give_up_s(settings)
            if give_up_s is not None and _clock() - started >= give_up_s:
                raise NoProgress(f"no answer from the server in "
                                 f"{settings.give_up_after_min:g} min (last: "
                                 f"{type(error).__name__})") from error
            failures += 1
            _wait(failures, settings, what, error)


def _check_space(path: Path, needed: int) -> None:
    done = path.with_name(path.name + ".part")
    needed -= done.stat().st_size if done.exists() else 0
    _require_space(path.parent, needed, path.name)


def _require_space(folder: Path, needed: int, what: str) -> None:
    free = shutil.disk_usage(folder).free
    if free < needed:
        raise DownloadError(f"{what}: needs {needed / 1e9:.1f} GB more, only "
                            f"{free / 1e9:.1f} GB free on {folder.anchor}")


def _run_workers(worker, count: int, settings: TransferSettings, last_byte_at) -> bool:
    """Run `worker` in `count` threads until they end, or until no new byte came for
    `give_up_after_min` (`last_byte_at()` says when the last one did): returns True then."""
    give_up_s = _give_up_s(settings)
    # Daemon threads: a stopped script, or a file set aside, must not wait for a connection
    # to time out.
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(count)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(0.5)
        if give_up_s is not None and _clock() - last_byte_at() >= give_up_s:
            return True
    return False


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
        settings, else from a `.part` without one (taken as the start of the file). A
        changed ETag or date alone, with the same size, doesn't discard the progress: some
        servers (e.g. behind redirects) change it, and a changed file of exactly the same
        size is unlikely."""
        if self.partial.exists() and self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if all(state.get(k) == v for k, v in self.key.items() if k != "validator"):
                if state.get("validator") != self.key["validator"]:
                    tqdm.write(f"{self.partial.name}: the server's ETag or date changed, the "
                               "size didn't: continuing. If unpacking fails, delete the .part "
                               "file to start over")
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
        initial = sum(self.done)
        self.progress_at = _clock()
        try:
            stuck = _run_workers(lambda: self._worker(todo), connections, self.settings,
                                 lambda: self.progress_at)
        finally:
            self.stop.set()
            with self.lock:
                self._save_state()
            self.bar.close()
        if self.errors:
            raise self.errors[0]
        if stuck:
            got = sum(self.done)
            raise NoProgress(f"no new byte in {self.settings.give_up_after_min:g} min; "
                             f"stopped at {got / 1e6:.1f} of {self.total / 1e6:.1f} MB "
                             f"({got / self.total:.1%})", gained=got - initial)
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
                        self.progress_at = now
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


class _ShareFolder:
    """The missing files of a share folder, fetched whole by several connections at once,
    one file each; see the module doc."""

    def __init__(self, files: list[_ShareFile], dest: Path, name: str,
                 settings: TransferSettings):
        self.files, self.dest, self.name, self.settings = files, dest, name, settings
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.errors: list[BaseException] = []  # these stop all connections
        self.failed: list[str] = []  # these stop only their own file

    def run(self) -> None:
        todo: queue.Queue[_ShareFile] = queue.Queue()
        total = have = 0
        for file in self.files:
            path = self.dest / file.path
            partial = path.with_name(path.name + ".part")
            total += file.size
            if path.exists() and path.stat().st_size == file.size:
                have += file.size
                continue
            if partial.exists() and partial.stat().st_size > file.size:
                partial.unlink()  # the file changed on the server
            have += partial.stat().st_size if partial.exists() else 0
            todo.put(file)
        if todo.empty():
            return
        self.dest.mkdir(parents=True, exist_ok=True)
        _require_space(self.dest, total - have, self.name)
        connections = max(1, min(self.settings.connections, todo.qsize()))
        tqdm.write(f"{self.name}: {len(self.files)} files, {total / 1e9:.2f} GB; "
                   f"{todo.qsize()} to fetch over {connections} connections")
        self.bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=self.name,
                        smoothing=0.05)
        self.progress_at = _clock()
        try:
            stuck = _run_workers(lambda: self._worker(todo), connections, self.settings,
                                 lambda: self.progress_at)
        finally:
            self.stop.set()
            self.bar.close()
        if self.errors:
            raise self.errors[0]
        if stuck:
            got = self.bar.n
            raise NoProgress(f"no new byte in {self.settings.give_up_after_min:g} min; "
                             f"stopped at {got / 1e6:.1f} of {total / 1e6:.1f} MB "
                             f"({got / total:.1%})", gained=got - have)
        if self.failed:
            raise DownloadError(f"{len(self.failed)} of {len(self.files)} files failed, e.g. "
                                + "; ".join(self.failed[:3]))

    def _worker(self, todo: "queue.Queue[_ShareFile]") -> None:
        session = _session(self.settings)
        try:
            while not self.stop.is_set():
                try:
                    file = todo.get_nowait()
                except queue.Empty:
                    return
                try:
                    self._fetch(session, file)
                except (DownloadError, requests.RequestException) as error:
                    problem = f"{file.path}: {type(error).__name__}: {error}"
                    tqdm.write(f"{self.name}/{problem}")
                    self.failed.append(problem)
        except BaseException as error:  # noqa: BLE001 - handed to the main thread
            self.errors.append(error)
            self.stop.set()
        finally:
            session.close()

    def _fetch(self, session: requests.Session, file: _ShareFile) -> None:
        path = self.dest / file.path
        partial = path.with_name(path.name + ".part")
        partial.parent.mkdir(parents=True, exist_ok=True)
        done = partial.stat().st_size if partial.exists() else 0
        failures = 0
        while done < file.size and not self.stop.is_set():
            offset = done
            try:
                done = self._stream(session, file, partial, offset)
                if done < file.size and not self.stop.is_set():
                    raise requests.ConnectionError("connection closed early")
            except Exception as error:  # sorted out below
                if _fatal(error) or not isinstance(error, requests.RequestException):
                    raise
                done = partial.stat().st_size if partial.exists() else 0
                failures = 0 if done > offset else failures
                failures += 1
                _wait(failures, self.settings, f"{self.name}/{file.path}", error, self.stop)
        if done == file.size:
            partial.touch()  # an empty file is never opened
            partial.replace(path)

    def _stream(self, session: requests.Session, file: _ShareFile, partial: Path,
                offset: int) -> int:
        """Add to `partial` from byte `offset` on; returns how far it got."""
        settings = self.settings
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(file.url, headers=headers, stream=True,
                         timeout=settings.timeout_s) as response:
            _check_response(response, file.url)
            got = response.headers.get("Content-Range", "")
            if response.status_code == 206 and not got.startswith(f"bytes {offset}-"):
                raise DownloadError(f"{file.url}: asked for bytes {offset}-, got {got!r}")
            if offset and response.status_code != 206:  # the whole file came: start over
                with self.lock:
                    self.bar.update(-offset)
                offset = 0
            # Unbuffered: what the progress bar counts is already with the OS, so `partial`
            # holds it after a crash of this script.
            with open(partial, "ab" if offset else "wb", buffering=0) as f:
                window_start, window_bytes = _clock(), 0
                for block in response.iter_content(READ_SIZE):
                    if self.stop.is_set():
                        break
                    block = block[:file.size - offset]
                    f.write(block)
                    offset += len(block)
                    window_bytes += len(block)
                    now = _clock()
                    with self.lock:
                        self.progress_at = now
                        self.bar.update(len(block))
                    if offset >= file.size:
                        break
                    if now - window_start >= settings.stall_s:
                        speed = window_bytes / (now - window_start) / 1000
                        if speed < settings.min_speed_kbs:
                            raise Stalled(f"{speed:.1f} kB/s over {settings.stall_s:g} s")
                        window_start, window_bytes = now, 0
        return offset


def _download_stream(url: str, partial: Path, total: int | None,
                     settings: TransferSettings) -> None:
    """One plain stream, for servers without range requests: a break restarts it."""
    session = _session(settings)
    failures, byte_at, give_up_s = 0, _clock(), _give_up_s(settings)
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
                            byte_at = _clock()
                            bar.update(len(block))
                if total is None or partial.stat().st_size == total:
                    return
                raise requests.ConnectionError("connection closed early")
            except Exception as error:  # sorted out below
                if _fatal(error) or not isinstance(error, requests.RequestException):
                    raise
                if give_up_s is not None and _clock() - byte_at >= give_up_s:
                    raise NoProgress(f"no new byte in {settings.give_up_after_min:g} min "
                                     "(this server restarts files from zero)") from error
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
