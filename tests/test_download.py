import io
import tarfile

import pytest
import requests

from weedseg.data import download

DATA = bytes(range(256)) * 40  # 10 kB


class FakeServer:
    """Stands in for requests.get; drops the connection after `drop_after` bytes, a given
    number of times."""

    def __init__(self, data, drop_after=None, drops=0, honour_range=True):
        self.data, self.drop_after, self.drops = data, drop_after, drops
        self.honour_range = honour_range
        self.ranges = []

    def get(self, url, headers=None, stream=True, timeout=None):
        start = int(headers["Range"][6:-1]) if headers and "Range" in headers else 0
        self.ranges.append(start)
        if not self.honour_range:
            start = 0
        drop = self.drops > 0
        self.drops -= drop
        return FakeResponse(self.data[start:], 206 if start else 200,
                            self.drop_after if drop else None)


class FakeResponse:
    def __init__(self, body, status_code, drop_after):
        self.body, self.status_code, self.drop_after = body, status_code, drop_after
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        sent = 0
        for i in range(0, len(self.body), 1000):
            if self.drop_after is not None and sent >= self.drop_after:
                raise requests.ConnectionError("Read timed out.")
            chunk = self.body[i:i + 1000]
            sent += len(chunk)
            yield chunk


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)


def test_download_resumes_after_dropped_connections(tmp_path, monkeypatch):
    server = FakeServer(DATA, drop_after=3000, drops=2)
    monkeypatch.setattr(download.requests, "get", server.get)
    target = tmp_path / "part.tar"

    download.download_file("https://x/part.tar", target)

    assert target.read_bytes() == DATA
    assert server.ranges == [0, 3000, 6000]  # each retry continues where the last stopped
    assert not (tmp_path / "part.tar.part").exists()


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    server = FakeServer(DATA, drop_after=3000, drops=1, honour_range=False)
    monkeypatch.setattr(download.requests, "get", server.get)
    target = tmp_path / "part.tar"
    download.download_file("https://x/part.tar", target)
    assert target.read_bytes() == DATA


def test_download_gives_up_after_max_retries(tmp_path, monkeypatch):
    server = FakeServer(DATA, drop_after=0, drops=100)
    monkeypatch.setattr(download.requests, "get", server.get)
    with pytest.raises(requests.ConnectionError):
        download.download_file("https://x/part.tar", tmp_path / "part.tar", max_retries=3)
    assert len(server.ranges) == 4


def test_download_and_extract_removes_archive(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("images/a.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    server = FakeServer(buffer.getvalue(), drop_after=2000, drops=1)
    monkeypatch.setattr(download.requests, "get", server.get)
    marker = tmp_path / ".downloaded" / "part"

    download.download_and_extract_tar("https://x/part.tar", tmp_path, marker)

    assert (tmp_path / "images" / "a.jpg").read_bytes() == b"abc"
    assert marker.exists()
    assert not (tmp_path / ".archives" / "part.tar").exists()
