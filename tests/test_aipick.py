"""--pick (aipick.py) and the OpenAI-compatible backend it and --copy can use."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from clipwright import aipick, copywriting
from clipwright.copywriting import CopyError, openai_json, write_copy
from clipwright.segment import Clip, Options, Signals
from clipwright.transcribe import Word


def signals(seconds: int = 600, loud: tuple[int, ...] = (100, 300, 500)) -> Signals:
    """Quiet video with a few loud bursts, so the signal pick has clear favourites."""
    loudness = np.zeros(seconds * 2)
    for t in loud:
        loudness[t * 2:(t + 20) * 2] = 2.0
    return Signals(hop=0.5, loudness=loudness)


def words_every_second(seconds: int = 600) -> list[Word]:
    return [Word(float(t), t + 0.5, f"w{t}") for t in range(seconds)]


def test_parse_keeps_valid_ratings_only():
    data = {"ratings": [{"n": 1, "score": 8, "why": "funny"}, {"n": 1, "score": 2, "why": "dup"},
                        {"n": 9, "score": 5, "why": "unknown"}, {"n": 2, "score": "x", "why": ""},
                        {"n": 3, "score": 14, "why": " big "}, {"n": True, "score": 3, "why": ""}, "junk"]}
    assert aipick.parse(data, 3) == {1: (8.0, "funny"), 3: (10.0, "big")}


@pytest.mark.parametrize("bad", [None, {}, {"ratings": "no"}, {"ratings": [{"n": 7, "score": 1, "why": ""}]}])
def test_parse_rejects(bad: Any):
    with pytest.raises(CopyError):
        aipick.parse(bad, 3)


def test_blend_lets_the_model_overrule_a_close_signal_call():
    clips = [Clip(0, 30, 3.0, ("loud moment",)), Clip(100, 130, 2.8), Clip(200, 230, 1.0)]
    out = aipick.blend(clips, {1: (1.0, "ad read"), 2: (9.0, "great story"), 3: (6.0, "")}, count=2)
    assert [c.start for c in out] == [100, 200]  # time order
    assert out[0].reasons[-1] == "AI 9/10: great story" and out[1].reasons[-1] == "AI 6/10"


def test_prompt_numbers_candidates_with_their_words():
    clips = [Clip(10, 13, 1.0), Clip(50, 52, 1.0)]
    p = aipick.prompt_for(clips, words_every_second(), "tiktok")
    assert "posted on tiktok" in p
    assert "#1 (3s)\nw10 w11 w12" in p and "#2 (2s)\nw50 w51" in p


def test_find_clips_uses_the_ratings():
    sig, words = signals(), words_every_second()
    opt = Options(count=2, min_len=20, max_len=30)
    asked: list[str] = []

    def ask(prompt: str, system: str, schema: dict[str, Any]) -> Any:
        asked.append(prompt)
        assert schema is aipick.SCHEMA and "standalone short" in system
        # the quiet candidates win on content: anything overlapping a loud burst gets 0
        starts = [int(m) for m in re.findall(r"\)\nw(\d+)", prompt)]
        loud = (100, 300, 500)
        return {"ratings": [{"n": i, "score": 0 if any(t - 30 < s < t + 20 for t in loud) else 10, "why": "x"}
                            for i, s in enumerate(starts, 1)]}

    plain = aipick.pick(aipick.candidates(sig, opt), opt)
    picked = aipick.find_clips(sig, opt, words, ask, log=lambda _m: None)
    assert len(asked) == 1 and len(picked) == 2
    assert {c.start for c in picked}.isdisjoint({c.start for c in plain})
    assert all(any(r.startswith("AI 10/10") for r in c.reasons) for c in picked)


def test_find_clips_falls_back_when_the_model_fails():
    sig, words = signals(), words_every_second()
    opt = Options(count=2, min_len=20, max_len=30)
    logs: list[str] = []

    def ask(*_a: Any) -> Any:
        raise CopyError("can't reach it")

    picked = aipick.find_clips(sig, opt, words, ask, log=logs.append)
    assert picked == aipick.pick(aipick.candidates(sig, opt), opt)
    assert any("AI pick failed (can't reach it)" in m for m in logs)


def test_find_clips_without_words_never_asks():
    def ask(*_a: Any) -> Any:
        raise AssertionError("asked")

    opt = Options(count=2, min_len=20, max_len=30)
    assert len(aipick.find_clips(signals(), opt, [], ask, log=lambda _m: None)) == 2


class Chat(BaseHTTPRequestHandler):
    reply: tuple[int, dict[str, Any]] = (200, {})
    seen: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append((self.path, dict(self.headers), body))
        code, out = self.reply
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_: Any) -> None:
        pass


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Chat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Chat.seen = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CLIPWRIGHT_OPENAI_MODEL", raising=False)
    monkeypatch.setenv("CLIPWRIGHT_OPENAI_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1/")
    yield
    srv.shutdown()
    srv.server_close()


def answer(content: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}]}


def test_openai_backend_writes_copy(chat: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    Chat.reply = (200, answer({"title": "T", "description": "D", "hashtags": ["a"]}))
    post = write_copy([Word(0, 1, "hello")], "openai", "shorts", "qwen3-8b")
    assert post.title == "T"
    path, headers, body = Chat.seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "qwen3-8b"
    assert body["response_format"]["json_schema"]["schema"] == copywriting.SCHEMA
    assert body["messages"][0]["role"] == "system" and "hello" in body["messages"][1]["content"]


def test_openai_without_a_key_sends_no_auth(chat: None):
    Chat.reply = (200, answer({"ratings": []}))
    assert openai_json("p", "s", {}, "m") == {"ratings": []}
    assert "Authorization" not in Chat.seen[0][1]


def test_openai_needs_a_model(chat: None):
    with pytest.raises(CopyError, match="--copy-model") as e:
        openai_json("p", "s", {})
    assert e.value.setup


def test_openai_model_from_the_environment(chat: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLIPWRIGHT_OPENAI_MODEL", "local-model")
    Chat.reply = (200, answer({}))
    openai_json("p", "s", {})
    assert Chat.seen[0][2]["model"] == "local-model"


@pytest.mark.parametrize(("code", "match", "setup"), [(401, "OPENAI_API_KEY", True), (404, "doesn't know model", True),
                                                      (500, "answered 500", False)])
def test_openai_errors(chat: None, code: int, match: str, setup: bool):
    Chat.reply = (code, {"error": "nope"})
    with pytest.raises(CopyError, match=match) as e:
        openai_json("p", "s", {}, "m")
    assert e.value.setup is setup


def test_openai_not_running(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(CopyError, match="is the server running") as e:
        openai_json("p", "s", {}, "m", url="http://127.0.0.1:9/v1")
    assert e.value.setup


def test_cli_rejects_pick_without_captions(tmp_path, capsys: pytest.CaptureFixture[str]):
    from clipwright.cli import main

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert main([str(video), "--pick", "claude", "--no-captions"]) == 2
    assert "--pick reads the transcript" in capsys.readouterr().err


def test_pipeline_uses_pick_and_records_it(tmp_path, monkeypatch: pytest.MonkeyPatch):
    from pathlib import Path

    from clipwright import pipeline

    calls: list[tuple[str, str | None]] = []

    def fake_ask(backend: str, prompt: str, system: str, schema: dict[str, Any], model: str | None = None) -> Any:
        calls.append((backend, model))
        return {"ratings": [{"n": 1, "score": 9, "why": "best"}]}

    monkeypatch.setattr(copywriting, "ask_json", fake_ask)
    info = type("I", (), {"path": tmp_path / "v.mp4", "duration": 600.0, "width": 1920, "height": 1080, "fps": 30.0})
    monkeypatch.setattr(pipeline, "probe", lambda p: info)
    monkeypatch.setattr(pipeline, "analyze", lambda *a: (signals(), words_every_second()))
    monkeypatch.setattr(pipeline, "_render_one",
                        lambda *a, **k: pipeline.Rendered(f"{a[5]}.mp4", None, None, None, a[3] - a[2]))
    monkeypatch.setattr(pipeline.transcribe, "save_words", lambda *a: None)
    job = pipeline.Job(source=Path("v.mp4"), out_dir=tmp_path, clips=Options(count=2, min_len=20, max_len=30),
                       pick="openai", copy_model="m")
    data = json.loads(pipeline.run(job, log=lambda _m: None).read_text())
    assert calls == [("openai", "m")]
    assert data["pick"] == "openai"
    assert all(any(r.startswith("AI ") for r in c["reasons"]) for c in data["clips"])
