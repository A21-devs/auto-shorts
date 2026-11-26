import asyncio
import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright import pipeline
from clipwright.mcp_server import ToolError, list_clips, make_clips, stage_progress


def _manifest(folder: Path) -> Path:
    folder.mkdir()
    m = folder / "manifest.json"
    m.write_text(json.dumps({
        "source": "C:/videos/talk.mp4", "platform": "shorts",
        "clips": [{"start": 1.0, "end": 21.5, "length": 20.5, "score": 0.8, "reasons": ["loud"],
                   "file": "talk_clip01.mp4", "thumbnail": "talk_clip01.jpg", "srt": None, "keep": True},
                  {"start": 40.0, "end": 60.0, "length": 20.0, "score": 0.5, "file": "talk_clip02.mp4"}],
    }), encoding="utf-8")
    return m


def test_list_clips_reads_a_manifest_or_its_folder(tmp_path: Path):
    m = _manifest(tmp_path / "talk_shorts")
    for arg in (str(m), str(m.parent), f'"{m}"'):
        out = list_clips(arg)
        assert out["platform"] == "shorts" and len(out["clips"]) == 2
    first, second = out["clips"]
    assert first["n"] == 1 and first["file"] == str(m.parent.resolve() / "talk_clip01.mp4")
    assert first["thumbnail"].endswith("talk_clip01.jpg") and first["srt"] is None and first["keep"] is True
    assert second["reasons"] == [] and second["thumbnail"] is None


def test_list_clips_errors(tmp_path: Path):
    with pytest.raises(ToolError, match="not found"):
        list_clips(str(tmp_path / "nope.json"))
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ToolError, match="isn't a clipwright manifest"):
        list_clips(str(tmp_path))


def test_make_clips_validates_before_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pytest.fail("should not run"))
    with pytest.raises(ToolError, match="not found"):
        make_clips(str(tmp_path / "nope.mp4"))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ToolError, match="count"):
        make_clips(str(video), count=0)
    with pytest.raises(ToolError, match="layout"):
        make_clips(str(video), layout="zoom")
    with pytest.raises(ToolError, match="links"):
        make_clips("https://www.youtube.com/watch?v=abc")


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_make_clips_runs_a_job_without_printing(tmp_path: Path, capsys):
    video = tmp_path / "talk.mp4"
    make_video(video, duration=40)
    out = make_clips(str(video), count=1, captions=False, platform=None, out=str(tmp_path / "out"))
    assert len(out["clips"]) == 1 and Path(out["clips"][0]["file"]).is_file()
    assert out["log"]
    # stdout is the MCP channel on stdio: the pipeline log must not end up there
    captured = capsys.readouterr()
    assert captured.out == "" and "rendered" in captured.err


def test_server_lists_tools_and_passes_errors_on(tmp_path: Path):
    pytest.importorskip("mcp")
    from mcp.server.mcpserver.exceptions import ToolError as SdkToolError

    from clipwright.mcp_server import build_server

    server = build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"make_clips", "list_clips"}
    with pytest.raises(SdkToolError, match="not found"):
        asyncio.run(server.call_tool("list_clips", {"manifest": str(tmp_path / "nope.json")}))


def test_parallel_calls_run_one_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    import threading
    import time

    from clipwright.mcp_server import build_server

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    active, peak = [0], [0]
    count = threading.Lock()

    def fake_run(job: pipeline.Job, log: pipeline.Log = print, progress: pipeline.Progress = print) -> Path:
        with count:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with count:
            active[0] -= 1
        out = tmp_path / f"out{time.perf_counter_ns()}"
        out.mkdir()
        (out / "manifest.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
        return out / "manifest.json"

    monkeypatch.setattr(pipeline, "run", fake_run)
    server = build_server()

    async def three() -> None:
        await asyncio.gather(*[server.call_tool("make_clips", {"video": str(video), "captions": False})
                               for _ in range(3)])

    asyncio.run(three())
    assert peak[0] == 1


def test_stage_progress_rises_through_the_stages():
    sent: list[tuple[float, str]] = []
    progress = stage_progress(lambda f, m: sent.append((f, m)))
    progress("audio", 0.0)
    progress("audio", 1.0)
    progress("scenes", 0.5)
    for i in range(1000):  # whisper reports per segment, most of these are too small to send
        progress("transcribe", i / 1000)
    progress("render", 0.5)
    progress("scenes", 1.0)  # out of order, must not go back
    progress("speaker", 0.3)  # unknown stage, ignored
    progress("done", 1.0)
    fractions = [f for f, _ in sent]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert len([m for _, m in sent if m.startswith("transcribe")]) <= 31
    assert ("done" in sent[-1][1]) and any(m == "render 50%" for _, m in sent)
    assert not any(m.startswith("speaker") for _, m in sent)


def _fake_run(tmp_path: Path, delay: float = 0.0):
    import time

    def fake_run(job: pipeline.Job, log: pipeline.Log = print,
                 progress: pipeline.Progress = lambda s, f: None) -> Path:
        for stage in ("audio", "scenes", "render"):
            progress(stage, 0.0)
            time.sleep(delay)
            progress(stage, 1.0)
        progress("done", 1.0)
        out = tmp_path / f"out{time.perf_counter_ns()}"
        out.mkdir()
        (out / "manifest.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
        return out / "manifest.json"

    return fake_run


def test_client_gets_progress_notifications(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    from mcp import Client

    from clipwright.mcp_server import build_server

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "run", _fake_run(tmp_path))
    got: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        got.append((progress, total, message))

    async def call() -> None:
        async with Client(build_server()) as client:
            result = await client.call_tool("make_clips", {"video": str(video), "captions": False},
                                            progress_callback=on_progress)
            assert not result.is_error

    asyncio.run(call())
    assert got and got[-1] == (1.0, 1.0, "done")
    assert [p for p, _, _ in got] == sorted(p for p, _, _ in got)
    assert any(m == "scenes 100%" for _, _, m in got)


def test_queued_call_says_it_is_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("mcp")
    from mcp import Client

    from clipwright.mcp_server import build_server

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "run", _fake_run(tmp_path, delay=0.1))
    messages: list[list[str | None]] = [[], []]
    values: list[list[float]] = [[], []]

    def collector(i: int):
        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            messages[i].append(message)
            values[i].append(progress)
        return on_progress

    async def two() -> None:
        async with Client(build_server()) as client:
            await asyncio.gather(*[client.call_tool("make_clips", {"video": str(video), "captions": False},
                                                    progress_callback=collector(i)) for i in range(2)])

    asyncio.run(two())
    waits = [m for m in messages if m and m[0] == "waiting for another make_clips call to finish"]
    assert len(waits) == 1 and all(m[-1] == "done" for m in messages)
    # the spec wants every notification above the last, including the first step after waiting
    assert all(all(a < b for a, b in zip(v, v[1:], strict=False)) for v in values)
