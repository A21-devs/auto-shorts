import asyncio
import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright import pipeline
from clipwright.mcp_server import ToolError, list_clips, make_clips


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

    def fake_run(job: pipeline.Job, log: pipeline.Log = print) -> Path:
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
