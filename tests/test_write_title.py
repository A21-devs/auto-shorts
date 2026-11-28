"""Writing a clip's title from the review page: pipeline.write_clip_copy and its endpoint."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_server import finished_job, request, srv, video  # noqa: F401 - fixtures

from clipwright import copywriting, server
from clipwright.copywriting import CopyError, PostCopy
from clipwright.pipeline import write_clip_copy
from clipwright.transcribe import Word, save_words


def finished_run(out: Path, copy: str | None = None, words: bool = True) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if words:
        save_words([Word(9.0, 9.5, "before"), Word(10.5, 11.0, "inside"), Word(14.0, 14.5, "too"),
                    Word(16.0, 16.5, "after")], out / "src_words.json")
    data = {"source": str(out / "src.mp4"), "duration": 60.0, "platform": "tiktok", "copy": copy,
            "copy_model": "qwen2.5:7b" if copy == "ollama" else None, "words": "src_words.json" if words else None,
            "clips": [{"start": 10.0, "end": 15.0, "file": "src_clip01.mp4"},
                      {"start": 30.0, "end": 35.0, "file": "src_clip02.mp4"}]}
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], str, str | None, str | None]]:
    calls: list[tuple[list[str], str, str | None, str | None]] = []

    def fake(words: list[Word], backend: str, platform: str | None, model: str | None) -> PostCopy:
        calls.append(([w.text for w in words], backend, platform, model))
        return PostCopy("A title", "A description.", ["one", "two"])

    monkeypatch.setattr(copywriting, "write_copy", fake)
    return calls


def test_writes_from_the_clip_range_with_the_runs_backend(tmp_path: Path, asked):
    manifest = finished_run(tmp_path, copy="ollama")
    entry = write_clip_copy(manifest, 1, log=lambda _m: None)
    assert asked == [(["inside", "too"], "ollama", "tiktok", "qwen2.5:7b")]
    assert entry["copy"] == {"title": "A title", "description": "A description.", "hashtags": ["one", "two"]}
    assert entry["post"] == "src_clip01.txt"
    assert (tmp_path / "src_clip01.txt").read_text(encoding="utf-8") == "A title\n\nA description.\n\n#one #two\n"
    saved = json.loads(manifest.read_text())["clips"]
    assert saved[0]["copy"]["title"] == "A title" and "copy" not in saved[1]


def test_backend_and_model_from_the_request_win(tmp_path: Path, asked):
    manifest = finished_run(tmp_path, copy="ollama")
    write_clip_copy(manifest, 2, "openai", "org/model", log=lambda _m: None)
    assert asked[0][1:] == ("openai", "tiktok", "org/model")


def test_the_runs_model_stays_with_the_runs_backend(tmp_path: Path, asked):
    manifest = finished_run(tmp_path, copy="ollama")  # run used qwen2.5:7b
    write_clip_copy(manifest, 1, "openai", None, log=lambda _m: None)  # a different backend, no model given
    write_clip_copy(manifest, 1, "ollama", None, log=lambda _m: None)  # the run's backend asked for by name
    assert [a[1:] for a in asked] == [("openai", "tiktok", None), ("ollama", "tiktok", "qwen2.5:7b")]


@pytest.mark.parametrize(("kwargs", "msg"), [
    ({"n": 3}, "doesn't exist"),
    ({"n": 1}, "choose a backend"),            # run didn't use --copy and none was given
    ({"n": 1, "backend": "gpt"}, "backend must be one of"),
])
def test_requests_that_cant_work(tmp_path: Path, asked, kwargs: dict[str, Any], msg: str):
    manifest = finished_run(tmp_path)
    with pytest.raises(ValueError, match=msg):
        write_clip_copy(manifest, log=lambda _m: None, **kwargs)
    assert asked == []


def test_no_transcript(tmp_path: Path, asked):
    manifest = finished_run(tmp_path, copy="claude", words=False)
    with pytest.raises(ValueError, match="no transcript|has none"):
        write_clip_copy(manifest, 1, log=lambda _m: None)


def test_a_failed_backend_keeps_the_old_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = finished_run(tmp_path, copy="claude")
    data = json.loads(manifest.read_text())
    data["clips"][0]["copy"] = {"title": "old", "description": "", "hashtags": []}
    manifest.write_text(json.dumps(data))

    def down(*_args: object) -> PostCopy:
        raise CopyError("can't reach Ollama", setup=True)

    monkeypatch.setattr(copywriting, "write_copy", down)
    with pytest.raises(CopyError):
        write_clip_copy(manifest, 1, log=lambda _m: None)
    assert json.loads(manifest.read_text())["clips"][0]["copy"]["title"] == "old"


# --- the endpoint ---

def test_endpoint_passes_backend_and_model(srv, video: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    job_id = finished_job(srv, video)
    calls = []

    def fake(manifest: Path, n: int, backend: str | None, model: str | None, log) -> dict[str, Any]:
        calls.append((n, backend, model))
        data = json.loads(manifest.read_text())
        data["clips"][n - 1]["copy"] = {"title": "t", "description": "d", "hashtags": []}
        manifest.write_text(json.dumps(data))
        return dict(data["clips"][n - 1])

    monkeypatch.setattr(server, "write_clip_copy", fake)
    status, _, body = request(srv, "POST", f"/api/jobs/{job_id}/clips/2", {"copy": "ollama", "copy_model": ""})
    assert status == 200, body
    assert json.loads(body)["manifest"]["clips"][1]["copy"]["title"] == "t"
    request(srv, "POST", f"/api/jobs/{job_id}/clips/1", {"copy": None})
    assert calls == [(2, "ollama", None), (1, None, None)]


@pytest.mark.parametrize(("body", "msg"), [
    ({"copy": "gpt"}, "copy must be one of"),
    ({"copy": "ollama", "copy_model": "a b"}, "copy_model"),
    ({"copy": None}, "choose a backend"),  # the fake run didn't use --copy
])
def test_endpoint_rejects(srv, video: Path, body: dict[str, Any], msg: str):  # noqa: F811
    job_id = finished_job(srv, video)
    status, _, data = request(srv, "POST", f"/api/jobs/{job_id}/clips/1", body)
    assert status == 400 and msg in json.loads(data)["error"]


def test_backend_failure_is_a_502(srv, video: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    job_id = finished_job(srv, video)

    def down(*_args: object, **_kw: object) -> dict[str, Any]:
        raise CopyError("can't reach Ollama at http://127.0.0.1:11434, is it running?", setup=True)

    monkeypatch.setattr(server, "write_clip_copy", down)
    status, _, data = request(srv, "POST", f"/api/jobs/{job_id}/clips/1", {"copy": "ollama"})
    assert status == 502 and "is it running" in json.loads(data)["error"]
    _, _, st = request(srv, "GET", f"/api/jobs/{job_id}")
    assert json.loads(st)["log"][-1].startswith("no title: can't reach Ollama")
