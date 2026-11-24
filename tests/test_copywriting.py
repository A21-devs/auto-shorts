import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from clipwright import copywriting
from clipwright.cli import main
from clipwright.copywriting import CopyError, PostCopy, ask_claude, ask_ollama, parse, prompt_for, write_copy
from clipwright.transcribe import Word

GOOD = {"title": "  He finally beat the boss  ", "description": "Two hours in.",
        "hashtags": ["#gaming", "boss fight", "Gaming", "", "eldenring"]}


def test_parse_tidies_hashtags():
    assert parse(GOOD) == PostCopy("He finally beat the boss", "Two hours in.", ["gaming", "bossfight", "eldenring"])


@pytest.mark.parametrize("bad", [None, [], {"title": "x"}, {"title": " ", "description": "d", "hashtags": []},
                                 {"title": "t", "description": "d", "hashtags": "a,b"}])
def test_parse_rejects(bad: Any):
    with pytest.raises(CopyError):
        parse(bad)


def test_prompt_mentions_the_platform_and_caps_length():
    words = [Word(i, i + 0.5, f"w{i}") for i in range(1000)]
    p = prompt_for(words, "shorts")
    assert "posted on shorts" in p and "w599" in p and "w600" not in p


def test_no_words_is_an_error_not_a_call():
    with pytest.raises(CopyError, match="no transcript"):
        write_copy([], "claude")


def fake_claude(tmp_path: Path, stdout: dict[str, Any]) -> list[str]:
    """A stand-in `claude` that records its stdin and prints a fixed --output-format json result."""
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import sys, json\n"
        f"open({str(tmp_path / 'stdin.txt')!r}, 'w', encoding='utf-8').write(sys.stdin.read())\n"
        f"print(json.dumps({stdout!r}))\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_claude_structured_output(tmp_path: Path):
    cmd = fake_claude(tmp_path, {"is_error": False, "result": "", "structured_output": GOOD})
    assert ask_claude("Transcript: hi", cmd).title == "He finally beat the boss"
    assert (tmp_path / "stdin.txt").read_text(encoding="utf-8") == "Transcript: hi"


def test_claude_not_signed_in(tmp_path: Path):
    cmd = fake_claude(tmp_path, {"is_error": True, "result": "Invalid API key · Please run /login to log in"})
    with pytest.raises(CopyError, match="claude auth login"):
        ask_claude("x", cmd)


def test_claude_missing():
    with pytest.raises(CopyError, match="isn't installed"):
        ask_claude("x", ("definitely-not-claude-xyz",))


class Ollama(BaseHTTPRequestHandler):
    reply: tuple[int, dict[str, Any]] = (200, {})
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        self.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        code, body = self.reply
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_: Any) -> None:
        pass


@pytest.fixture
def ollama():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Ollama.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/api/generate"
    srv.shutdown()
    srv.server_close()


def test_ollama(ollama: str):
    Ollama.reply = (200, {"response": json.dumps(GOOD)})
    assert ask_ollama("p", "qwen3", url=ollama).hashtags == ["gaming", "bossfight", "eldenring"]
    sent = Ollama.seen[0]
    assert sent["model"] == "qwen3" and sent["format"] == copywriting.SCHEMA and sent["stream"] is False


def test_ollama_missing_model(ollama: str):
    Ollama.reply = (404, {"error": "model 'qwen3' not found"})
    with pytest.raises(CopyError, match="ollama pull qwen3"):
        ask_ollama("p", "qwen3", url=ollama)


def test_ollama_not_running():
    with pytest.raises(CopyError, match="is it running"):
        ask_ollama("p", url="http://127.0.0.1:9/api/generate")


def test_cli_copy_needs_captions(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert main([str(video), "--copy", "claude", "--no-captions"]) == 2
    assert "--no-captions" in capsys.readouterr().err


def test_pipeline_keeps_going_when_copy_fails(monkeypatch: pytest.MonkeyPatch):
    from clipwright import pipeline

    def boom(*_a: Any) -> PostCopy:
        raise CopyError("can't reach Ollama")

    monkeypatch.setattr(copywriting, "write_copy", boom)
    logs: list[str] = []
    job = pipeline.Job(source=Path("v.mp4"), copy="ollama")
    assert pipeline._post_copy(job, [Word(1, 2, "hi")], 0, 5, logs.append) == (None, True)
    assert logs == ["         no title/hashtags: can't reach Ollama"]


def test_pipeline_only_sends_the_clip_words(monkeypatch: pytest.MonkeyPatch):
    from clipwright import pipeline
    seen: list[list[str]] = []

    def fake(words: list[Word], backend: str, platform: str | None, model: str) -> PostCopy:
        seen.append([w.text for w in words])
        return PostCopy("t", "d", ["x"])

    monkeypatch.setattr(copywriting, "write_copy", fake)
    job = pipeline.Job(source=Path("v.mp4"), copy="claude", platform="tiktok")
    words = [Word(1, 2, "before"), Word(10, 11, "inside"), Word(30, 31, "after")]
    assert pipeline._post_copy(job, words, 5, 20, lambda _m: None) == (
        {"title": "t", "description": "d", "hashtags": ["x"]}, True)
    assert seen == [["inside"]]


@pytest.mark.parametrize(("setup", "calls"), [(True, 1), (False, 3)])
def test_setup_failures_stop_after_the_first_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: bool,
                                                  calls: int):
    from clipwright import pipeline
    from clipwright.segment import Clip
    tries: list[int] = []

    def fail(*_a: Any) -> PostCopy:
        tries.append(1)
        raise CopyError("nope", setup=setup)

    monkeypatch.setattr(copywriting, "write_copy", fail)
    monkeypatch.setattr(pipeline, "probe", lambda p: type("I", (), {"path": Path("v.mp4"), "duration": 90.0,
                                                                    "width": 1, "height": 1, "fps": 30.0})())
    monkeypatch.setattr(pipeline, "analyze", lambda *a: (None, [Word(1, 2, "hi")]))
    monkeypatch.setattr(pipeline.segment, "find_clips",
                        lambda *a: [Clip(i * 20.0, i * 20.0 + 15, 1.0, ()) for i in range(3)])
    monkeypatch.setattr(pipeline, "_render_one",
                        lambda *a, **k: pipeline.Rendered("c.mp4", None, None, None, 15.0))
    monkeypatch.setattr(pipeline.transcribe, "save_words", lambda *a: None)
    job = pipeline.Job(source=Path("v.mp4"), out_dir=tmp_path, copy="claude")
    data = json.loads(pipeline.run(job, log=lambda _m: None).read_text())
    assert len(tries) == calls
    assert [c["copy"] for c in data["clips"]] == [None, None, None]
