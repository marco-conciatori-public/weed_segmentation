import hashlib
import io
import itertools
import json
import tarfile
import threading
import time
import zipfile

import pytest
import requests

from weedseg.data import download
from weedseg.data.download import DownloadError, NoProgress, TransferSettings

DATA = bytes(range(256)) * 40  # 10 kB
CHUNK = 2048  # bytes per piece in these tests: DATA is 5 pieces
BLOCK = 1000  # bytes per read from the fake connection


def settings(**changes):
    return TransferSettings(**{"connections": 3, "chunk_mb": CHUNK / (1 << 20),
                               "max_retries": 3, "give_up_after_min": None, **changes})


# Gives up after 0.03 s without progress, retrying every 0.01 s meanwhile.
QUICK_GIVE_UP = {"give_up_after_min": 0.0005, "max_retries": None}


class FakeServer:
    """Stands in for requests.Session: serves `data` with byte ranges (unless
    `honour_range` is False), drops the next `drops` connections after `drop_after` bytes,
    fails every request once `budget` bytes have been served (the network is gone), and
    resets every request for bytes from `dead_from` on (a server that can't serve them)."""

    def __init__(self, data, drop_after=None, drops=0, honour_range=True, budget=None,
                 status=200, content_type="application/zip", dead_from=None):
        self.data, self.drop_after, self.drops = data, drop_after, drops
        self.honour_range, self.budget, self.dead_from = honour_range, budget, dead_from
        self.status, self.content_type = status, content_type
        self.lock = threading.Lock()
        self.requests = []  # (start, end) of each request; None without a range
        self.served = 0

    def session(self):
        return FakeSession(self)

    def respond(self, headers, url=None):
        with self.lock:
            if self.budget is not None and self.served >= self.budget:
                raise requests.ConnectionError("network down")
            ranged = "Range" in headers and self.honour_range
            start, end = 0, len(self.data) - 1
            if ranged:
                first, last = headers["Range"][len("bytes="):].split("-")
                start, end = int(first), int(last) if last else end
            self.requests.append((start, end) if ranged else None)
            if self.dead_from is not None and start >= self.dead_from:
                raise requests.ConnectionError("connection reset")
            drop = None
            if self.drops and end > start:  # never the 1-byte probe
                self.drops -= 1
                drop = self.drop_after
        body = self.data[start:end + 1]
        headers = {"Content-Length": str(len(body)), "Content-Type": self.content_type}
        if ranged:
            headers["Content-Range"] = f"bytes {start}-{end}/{len(self.data)}"
        return FakeResponse(self, body, 206 if ranged else self.status, headers, drop)


class FakeRoutes:
    """Several fake servers, one per URL."""

    def __init__(self, servers):
        self.servers = servers

    def session(self):
        return FakeSession(self)

    def respond(self, headers, url):
        return self.servers[url].respond(headers)


class FakeSession:
    def __init__(self, server):
        self.server, self.proxies = server, {}

    def get(self, url, headers=None, stream=True, timeout=None):
        return self.server.respond(headers or {}, url)

    def close(self):
        pass


class FakeResponse:
    def __init__(self, server, body, status_code, headers, drop_after):
        self.server, self.body, self.status_code = server, body, status_code
        self.headers, self.drop_after = headers, drop_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def iter_content(self, chunk_size):
        sent = 0
        for i in range(0, len(self.body), BLOCK):
            if self.drop_after is not None and sent >= self.drop_after:
                raise requests.ConnectionError("Read timed out.")
            block = self.body[i:i + BLOCK]
            with self.server.lock:
                if self.server.budget is not None:
                    block = block[:max(0, self.server.budget - self.server.served)]
                self.server.served += len(block)
            if not block:
                raise requests.ConnectionError("network down")
            sent += len(block)
            yield block


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(download, "_sleep", lambda seconds, stop=None: None)


@pytest.fixture
def short_sleep(monkeypatch):
    """Retries every 0.01 s, for tests that wait for the give-up time."""
    monkeypatch.setattr(download, "_sleep", lambda seconds, stop=None: time.sleep(0.01))


def serve(monkeypatch, server):
    monkeypatch.setattr(download.requests, "Session", server.session)
    return server


def test_pieces_over_several_connections_resume_after_drops(tmp_path, monkeypatch):
    server = serve(monkeypatch, FakeServer(DATA, drop_after=BLOCK, drops=3))
    target = tmp_path / "data.zip"
    download.download_file("https://x/data.zip", target, settings())
    assert target.read_bytes() == DATA
    assert server.served == len(DATA)  # every retry continued where the last one stopped
    assert len(server.requests) == 1 + 5 + 3  # probe, pieces, retries
    assert not list(tmp_path.glob("data.zip.part*"))


def test_a_stopped_run_continues_from_its_saved_progress(tmp_path, monkeypatch):
    target = tmp_path / "data.zip"
    serve(monkeypatch, FakeServer(DATA, budget=4500))
    with pytest.raises(requests.ConnectionError):
        download.download_file("https://x/data.zip", target, settings(connections=1,
                                                                      max_retries=1))
    state = json.loads((tmp_path / "data.zip.part.json").read_text())
    assert sum(state["done"]) == 4500
    server = serve(monkeypatch, FakeServer(DATA))
    download.download_file("https://x/data.zip", target, settings())
    assert target.read_bytes() == DATA
    assert server.served == len(DATA) - 4500


def test_an_unfinished_file_without_progress_is_continued(tmp_path, monkeypatch):
    (tmp_path / "data.zip.part").write_bytes(DATA[:3000])  # e.g. from a browser
    server = serve(monkeypatch, FakeServer(DATA))
    download.download_file("https://x/data.zip", tmp_path / "data.zip", settings())
    assert (tmp_path / "data.zip").read_bytes() == DATA
    assert server.served == len(DATA) - 3000


def test_checksums_are_checked(tmp_path, monkeypatch):
    serve(monkeypatch, FakeServer(DATA))
    md5 = hashlib.md5(DATA).hexdigest()
    download.download_file("https://x/a.zip", tmp_path / "a.zip", settings(), md5=md5)
    assert (tmp_path / "a.zip").exists()
    with pytest.raises(DownloadError, match="MD5"):
        download.download_file("https://x/b.zip", tmp_path / "b.zip", settings(), md5="0" * 32)
    assert (tmp_path / "b.zip.bad").exists() and not (tmp_path / "b.zip").exists()
    assert not (tmp_path / "b.zip.part.json").exists()  # the next run starts over


def test_a_stalled_connection_is_reopened(tmp_path, monkeypatch):
    clock = itertools.count(step=100)  # 100 s pass between reads: 0.01 kB/s
    monkeypatch.setattr(download, "_clock", lambda: next(clock))
    server = serve(monkeypatch, FakeServer(DATA))
    download.download_file("https://x/data.zip", tmp_path / "data.zip",
                           settings(connections=1, stall_s=50))
    assert (tmp_path / "data.zip").read_bytes() == DATA
    # every read but a piece's last one stalls: reads of 1000, 1000 and 48 bytes, so each
    # piece takes three connections, none of them wasted once the piece is complete
    assert len(server.requests) == 1 + 3 * 5


def test_a_server_without_ranges_gets_one_restarted_stream(tmp_path, monkeypatch):
    serve(monkeypatch, FakeServer(DATA, drop_after=3000, drops=1, honour_range=False))
    download.download_file("https://x/data.zip", tmp_path / "data.zip", settings())
    assert (tmp_path / "data.zip").read_bytes() == DATA


@pytest.mark.parametrize("server, error", [
    (FakeServer(DATA, content_type="text/html; charset=utf-8"), "web page"),
    (FakeServer(DATA, status=404, honour_range=False), "404"),
])
def test_errors_a_retry_cannot_fix_stop_at_once(tmp_path, monkeypatch, server, error):
    serve(monkeypatch, server)
    with pytest.raises((DownloadError, requests.HTTPError), match=error):
        download.download_file("https://x/data.zip", tmp_path / "data.zip", settings())
    assert len(server.requests) == 1


def test_a_file_that_changed_on_the_server_is_noticed(tmp_path, monkeypatch):
    serve(monkeypatch, FakeServer(DATA))
    with pytest.raises(DownloadError, match="changed"):
        download.download_file("https://x/data.zip", tmp_path / "data.zip", settings(),
                               size=123)


def archive(kind):
    buffer = io.BytesIO()
    if kind == "zip":
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("images/a.jpg", b"abc")
    else:
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            info = tarfile.TarInfo("images/a.jpg")
            info.size = 3
            tar.addfile(info, io.BytesIO(b"abc"))
    return buffer.getvalue()


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_fetch_files_unpacks_archives_and_skips_finished_ones(tmp_path, monkeypatch, kind):
    files = [{"url": f"https://x/part.{kind}", "extract": True}]
    server = serve(monkeypatch, FakeServer(archive(kind)))
    download.fetch_files(files, tmp_path, settings())
    assert (tmp_path / "images" / "a.jpg").read_bytes() == b"abc"
    assert (tmp_path / ".downloaded" / "part").exists()
    assert not (tmp_path / ".archives" / f"part.{kind}").exists()
    requests_made = len(server.requests)
    download.fetch_files(files, tmp_path, settings())
    assert len(server.requests) == requests_made


def test_a_file_without_progress_is_set_aside_with_its_progress_kept(tmp_path, monkeypatch,
                                                                     short_sleep):
    target = tmp_path / "data.zip"
    serve(monkeypatch, FakeServer(DATA, dead_from=2 * CHUNK))  # serves only 2 pieces of 5
    with pytest.raises(NoProgress, match="stopped at 0.0 of 0.0 MB") as caught:
        download.download_file("https://x/data.zip", target, settings(**QUICK_GIVE_UP))
    assert caught.value.gained == 2 * CHUNK
    state = json.loads((tmp_path / "data.zip.part.json").read_text())
    assert state["done"] == [CHUNK, CHUNK, 0, 0, 0]
    server = serve(monkeypatch, FakeServer(DATA))  # the server is mended
    download.download_file("https://x/data.zip", target, settings())
    assert target.read_bytes() == DATA
    assert server.served == len(DATA) - 2 * CHUNK


def test_stuck_and_failing_files_dont_stop_the_others(tmp_path, monkeypatch, short_sleep):
    servers = {"https://x/a.zip": FakeServer(DATA, dead_from=0),  # down, even the probe
               "https://x/c.zip": FakeServer(DATA, status=404, honour_range=False),
               "https://x/b.zip": FakeServer(DATA)}
    serve(monkeypatch, FakeRoutes(servers))
    files = [{"url": url} for url in servers]
    progress, unfinished = download.fetch_files(files, tmp_path, settings(**QUICK_GIVE_UP))
    assert progress and (tmp_path / "b.zip").read_bytes() == DATA
    assert [(item.name, item.retry) for item in unfinished] == [("a.zip", True),
                                                                ("c.zip", False)]
    assert "no answer from the server" in unfinished[0].reason


def test_set_aside_files_get_passes_only_while_something_comes_in(tmp_path, monkeypatch,
                                                                  short_sleep):
    servers = {"https://x/a.zip": FakeServer(DATA, dead_from=0),
               "https://x/c.zip": FakeServer(DATA, status=404, honour_range=False),
               "https://x/b.zip": FakeServer(DATA)}
    serve(monkeypatch, FakeRoutes(servers))
    passes = []
    fetch_files = download.fetch_files
    monkeypatch.setattr(download, "fetch_files",
                        lambda *args, **kwargs: passes.append(1) or fetch_files(*args, **kwargs))
    datasets = [("one", {"files": [{"url": url} for url in servers]}, tmp_path)]
    with pytest.raises(DownloadError, match=r"c\.zip: HTTPError[\s\S]*a\.zip: no answer"):
        download.fetch_all(datasets, settings(**QUICK_GIVE_UP))
    # b.zip came in on the first pass, so a.zip got a second one, which brought nothing
    assert len(passes) == 2
    assert len(servers["https://x/c.zip"].requests) == 1  # a 404 is not retried
    assert (tmp_path / "b.zip").exists()


def test_unknown_transfer_settings_are_refused():
    with pytest.raises(ValueError, match="conections"):
        TransferSettings.from_config({"conections": 4})
