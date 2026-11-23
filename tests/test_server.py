import http.client
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autoshorts import server
from autoshorts.pipeline import Job
from autoshorts.server import App, BadRequest, make_server, parse_job, parse_range

release = threading.Event()


def fake_run(job: Job, log, progress) -> Path:
    """Stands in for the real pipeline: two fake clips and a manifest, with progress along the way."""
    if job.source.name == "broken.mp4":
        raise RuntimeError("ffmpeg could not read the audio")
    if job.source.name == "slow.mp4":
        release.wait(5)
    log("reading audio")
    progress("audio", 1.0)
    job.out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in (1, 2):
        name = f"my clip_clip{i:02d}.mp4"
        (job.out_dir / name).write_bytes(bytes(range(256)) * 4)  # 1024 bytes
        clips.append({"start": i * 10.0, "end": i * 10.0 + 5, "file": name})
        progress("render", i / 2)
    manifest = job.out_dir / "manifest.json"
    manifest.write_text(json.dumps({"clips": clips}))
    progress("done", 1.0)
    return manifest


@pytest.fixture
def srv(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, int]]:
    monkeypatch.setattr(server, "run", fake_run)
    s = make_server(port=0, app=App())
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    yield "127.0.0.1", s.server_address[1]
    s.shutdown()
    s.server_close()


def request(addr: tuple[str, int], method: str, path: str, body: Any = None,
            headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection(*addr, timeout=5)
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    out = resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    conn.close()
    return out


def wait_done(addr: tuple[str, int], job_id: str) -> dict[str, Any]:
    for _ in range(100):
        _, _, body = request(addr, "GET", f"/api/jobs/{job_id}")
        st = json.loads(body)
        if st["state"] in ("done", "error"):
            return st  # type: ignore[no-any-return]
        time.sleep(0.05)
    raise AssertionError("job never finished")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    v = tmp_path / "my clip.mp4"
    v.write_bytes(b"x")
    return v


# --- parse_job ---

def test_parse_job_defaults(video: Path):
    job = parse_job({"video": f'"{video}"'})  # quotes from Explorer's "Copy as path"
    assert job.source == video.resolve()
    assert job.out_dir == video.resolve().with_name("my clip_shorts")
    assert (job.clips.count, job.clips.min_len, job.clips.max_len) == (6, 20, 55)
    assert job.render.layout == "blur" and job.captions and job.whisper_model == "small"


@pytest.mark.parametrize(("patch", "msg"), [
    ({"video": None}, "video must be"),
    ({"video": "nope.mp4"}, "not found"),
    ({"count": 0}, "count"),
    ({"count": 2.5}, "whole number"),
    ({"count": True}, "count"),
    ({"min": 30, "max": 20}, "max must be"),
    ({"layout": "stretch"}, "layout"),
    ({"captions": "yes"}, "captions"),
    ({"model": "../../evil"}, "model"),
    ({"out": 5}, "out"),
])
def test_parse_job_rejects(video: Path, patch: dict[str, Any], msg: str):
    with pytest.raises(BadRequest, match=msg):
        parse_job({"video": str(video), **patch})


# --- ranges ---

@pytest.mark.parametrize(("header", "expected"), [
    ("bytes=0-99", (0, 99)),
    ("bytes=100-", (100, 1023)),
    ("bytes=-24", (1000, 1023)),
    ("bytes=1000-5000", (1000, 1023)),
    ("bytes=-5000", (0, 1023)),
    ("bytes=1024-", None),
    ("bytes=50-10", None),
    ("bytes=-", None),
    ("bytes=-0", None),
    ("items=0-1", None),
    ("bytes=0-1,5-6", None),
])
def test_parse_range(header: str, expected: tuple[int, int] | None):
    assert parse_range(header, 1024) == expected


# --- HTTP ---

def test_job_lifecycle_and_files(srv, video: Path):
    status, _, body = request(srv, "POST", "/api/jobs", {"video": str(video), "count": 2})
    assert status == 201
    job_id = json.loads(body)["id"]

    st = wait_done(srv, job_id)
    assert st["state"] == "done", st
    assert (st["stage"], st["fraction"]) == ("done", 1.0)
    assert st["log"] == ["reading audio"]
    assert [c["file"] for c in st["manifest"]["clips"]] == ["my clip_clip01.mp4", "my clip_clip02.mp4"]

    # file names with spaces come URL-encoded from the browser
    status, headers, data = request(srv, "GET", f"/api/jobs/{job_id}/files/my%20clip_clip01.mp4")
    assert status == 200 and len(data) == 1024
    assert headers["content-type"] == "video/mp4" and headers["accept-ranges"] == "bytes"

    status, headers, data = request(srv, "GET", f"/api/jobs/{job_id}/files/my%20clip_clip01.mp4",
                                    headers={"Range": "bytes=256-511"})
    assert status == 206 and data == bytes(range(256))
    assert headers["content-range"] == "bytes 256-511/1024"

    status, headers, _ = request(srv, "GET", f"/api/jobs/{job_id}/files/my%20clip_clip01.mp4",
                                 headers={"Range": "bytes=4096-"})
    assert status == 416 and headers["content-range"] == "bytes */1024"

    status, _, body = request(srv, "GET", "/api/jobs")
    assert [j["id"] for j in json.loads(body)] == [job_id]


def test_pipeline_error_is_reported(srv, tmp_path: Path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"x")
    _, _, body = request(srv, "POST", "/api/jobs", {"video": str(broken)})
    st = wait_done(srv, json.loads(body)["id"])
    assert st["state"] == "error"
    assert st["error"] == "ffmpeg could not read the audio"
    assert "RuntimeError" in st["log"][-1]


@pytest.mark.parametrize("name", ["..%2Fsecret.txt", "..", "%2E%2E", "..%5Csecret.txt", "C%3Asecret.txt",
                                  "missing.mp4"])
def test_files_outside_the_output_folder_are_refused(srv, video: Path, name: str):
    (video.parent / "secret.txt").write_text("nope")
    _, _, body = request(srv, "POST", "/api/jobs", {"video": str(video)})
    job_id = json.loads(body)["id"]
    wait_done(srv, job_id)
    status, _, data = request(srv, "GET", f"/api/jobs/{job_id}/files/{name}")
    assert status == 404
    assert b"nope" not in data


def test_post_needs_json_content_type(srv, video: Path):
    # what a cross-origin <form> post would look like
    status, _, _ = request(srv, "POST", "/api/jobs", f"video={video}".encode(),
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 415


@pytest.mark.parametrize(("body", "status"), [(b"{not json", 400), (b"[1, 2]", 400), (b"x" * 70_000, 413)])
def test_bad_bodies(srv, body: bytes, status: int):
    assert request(srv, "POST", "/api/jobs", body)[0] == status


def test_negative_content_length_is_400_not_a_hang(srv):
    with socket.create_connection(srv, timeout=3) as c:
        c.sendall(b"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: -1\r\n\r\n{}")
        assert c.recv(64).startswith(b"HTTP/1.0 400")


def test_one_active_job_per_output_folder(srv, tmp_path: Path):
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(b"x")
    release.clear()
    try:
        status, _, body = request(srv, "POST", "/api/jobs", {"video": str(slow)})
        first = json.loads(body)["id"]
        status, _, body = request(srv, "POST", "/api/jobs", {"video": str(slow), "count": 3})
        assert status == 409 and first in json.loads(body)["error"]
        # a different folder is fine
        status, _, _ = request(srv, "POST", "/api/jobs", {"video": str(slow), "out": str(tmp_path / "other")})
        assert status == 201
    finally:
        release.set()
    assert wait_done(srv, first)["state"] == "done"

    # once it's finished, a new run may reuse the folder, and the old job says it was replaced
    status, _, body = request(srv, "POST", "/api/jobs", {"video": str(slow)})
    assert status == 201
    second = json.loads(body)["id"]
    _, _, body = request(srv, "GET", f"/api/jobs/{first}")
    assert json.loads(body)["superseded_by"] == second
    wait_done(srv, second)


def test_job_list_has_no_logs(srv, video: Path):
    _, _, body = request(srv, "POST", "/api/jobs", {"video": str(video)})
    wait_done(srv, json.loads(body)["id"])
    _, _, body = request(srv, "GET", "/api/jobs")
    assert json.loads(body)[0]["log"] == []


def test_bad_job_options_are_400(srv):
    status, _, body = request(srv, "POST", "/api/jobs", {"video": "does/not/exist.mp4"})
    assert status == 400 and "not found" in json.loads(body)["error"]


@pytest.mark.parametrize("host", ["evil.example:8765", "192.168.1.5:8765", ""])
def test_foreign_host_header_is_refused(srv, host: str):
    # DNS rebinding: a page on evil.example resolves to 127.0.0.1 but the browser still sends its own Host
    assert request(srv, "GET", "/api/jobs", headers={"Host": host})[0] == 403


@pytest.mark.parametrize("host", ["localhost:1234", "127.0.0.1", "[::1]:8765"])
def test_local_host_headers_are_fine(srv, host: str):
    assert request(srv, "GET", "/api/jobs", headers={"Host": host})[0] == 200


def test_unknown_routes_and_jobs(srv):
    assert request(srv, "GET", "/api/nope")[0] == 404
    assert request(srv, "GET", "/api/jobs/doesnotexist")[0] == 404
    assert request(srv, "GET", "/api/jobs/doesnotexist/files/a.mp4")[0] == 404
