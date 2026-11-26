import asyncio
import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright import download, pipeline
from clipwright.mcp_server import LINK_SPAN, RunProgress, ToolError, list_clips, make_clips, rerender_clip


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
    with pytest.raises(ToolError, match="layout must be one of blur, crop, follow, speaker, fit"):
        make_clips(str(video), layout="zoom")
    monkeypatch.setattr(download, "available", lambda: False)
    with pytest.raises(ToolError, match=r"links need yt-dlp: pip install \"clipwright\[link\]\""):
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
    assert names == {"make_clips", "list_clips", "rerender_clip"}
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


def test_run_progress_rises_through_the_stages():
    sent: list[tuple[float, str]] = []
    progress = RunProgress(lambda f, m: sent.append((f, m)))
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


def test_note_stays_above_the_last_step():
    sent: list[float] = []
    progress = RunProgress(lambda f, m: sent.append(f), LINK_SPAN)
    progress.note("waiting")  # nothing sent yet: starts at 0
    progress("download", 0.5)
    progress("download", 1.0)
    progress.note("waiting")
    progress("audio", 0.0)  # same value as the end of the download, dropped
    progress("audio", 1.0)
    progress("done", 1.0)
    assert sent[0] == 0.0 and sent[2] == pytest.approx(0.3)
    assert all(a < b for a, b in zip(sent, sent[1:], strict=False)) and sent[-1] == 1.0
    assert len(sent) == 6


def _fake_download(tmp_path: Path, calls: list[dict], srt: bool = True):
    def fake(url: str, folder: Path, log=print, on_progress=None, subtitles=None) -> Path:
        calls.append({"url": url, "folder": folder, "subtitles": subtitles})
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "Talk [abc].mp4"
        video.write_bytes(b"x")
        if srt and subtitles:
            video.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
        for f in (0.0, 0.5, 1.0):
            if on_progress:
                on_progress(f)
        log(f"downloading {url}")
        return video

    return fake


def test_make_clips_downloads_a_link_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    jobs: list[pipeline.Job] = []
    monkeypatch.setattr(download, "available", lambda: True)
    monkeypatch.setattr(download, "download", _fake_download(tmp_path, calls))
    fake_run = _fake_run(tmp_path)

    def run(job: pipeline.Job, log: pipeline.Log = print, progress: pipeline.Progress = lambda s, f: None) -> Path:
        jobs.append(job)
        return fake_run(job, log, progress)

    monkeypatch.setattr(pipeline, "run", run)
    sent: list[tuple[float, str]] = []
    out = make_clips(" https://www.youtube.com/watch?v=abc ", downloads=str(tmp_path / "dl"),
                     report=lambda f, m: sent.append((f, m)))
    assert calls == [{"url": "https://www.youtube.com/watch?v=abc", "folder": tmp_path / "dl", "subtitles": "en"}]
    job = jobs[0]
    assert job.source == tmp_path / "dl" / "Talk [abc].mp4"
    assert job.srt == job.source.with_suffix(".srt")
    assert job.out_dir == tmp_path / "dl" / "Talk [abc]_shorts"
    assert job.url == "https://www.youtube.com/watch?v=abc"
    assert sent[0] == (0.0, "download 0%") and ("download" in sent[2][1]) and sent[-1] == (1.0, "done")
    fractions = [f for f, _ in sent]
    assert fractions == sorted(fractions)
    assert out["log"][0] == "downloading https://www.youtube.com/watch?v=abc"


def test_link_without_captions_skips_subtitles_and_keeps_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []
    jobs: list[pipeline.Job] = []
    monkeypatch.setattr(download, "available", lambda: True)
    monkeypatch.setattr(download, "download", _fake_download(tmp_path, calls))
    fake_run = _fake_run(tmp_path)
    monkeypatch.setattr(pipeline, "run", lambda job, log=print, progress=None: jobs.append(job) or fake_run(job))
    monkeypatch.setattr("clipwright.mcp_server.DEFAULT_DOWNLOADS", tmp_path / "default")
    make_clips("https://www.twitch.tv/videos/1", captions=False, out=str(tmp_path / "mine"))
    assert calls[0]["subtitles"] is None and calls[0]["folder"] == tmp_path / "default"
    assert jobs[0].srt is None and jobs[0].out_dir == (tmp_path / "mine").resolve()


def test_download_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(download, "available", lambda: True)

    def broken(url: str, folder: Path, **_: object) -> Path:
        raise download.DownloadError("Video unavailable. This video is private")

    monkeypatch.setattr(download, "download", broken)
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pytest.fail("should not run"))
    with pytest.raises(download.DownloadError, match="private"):
        make_clips("https://www.youtube.com/watch?v=abc", downloads=str(tmp_path))


def test_downloads_into_one_folder_run_one_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import threading
    import time

    monkeypatch.setattr(download, "available", lambda: True)
    active: dict[Path, int] = {}
    peak: dict[Path, int] = {}
    overall = [0, 0]
    guard = threading.Lock()

    def slow(url: str, folder: Path, log=print, on_progress=None, subtitles=None) -> Path:
        with guard:
            active[folder] = active.get(folder, 0) + 1
            peak[folder] = max(peak.get(folder, 0), active[folder])
            overall[0] += 1
            overall[1] = max(overall[1], overall[0])
        time.sleep(0.2)
        with guard:
            active[folder] -= 1
            overall[0] -= 1
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / "v.mp4"
        video.write_bytes(b"x")
        return video

    monkeypatch.setattr(download, "download", slow)
    monkeypatch.setattr(pipeline, "run", _fake_run(tmp_path))
    notes: list[list[str]] = [[] for _ in range(4)]
    folders = [tmp_path / "a", tmp_path / "a", tmp_path / "a", tmp_path / "b"]
    threads = [threading.Thread(target=make_clips, args=("https://example.com/v",),
                                kwargs={"downloads": str(f), "captions": False,
                                        "out": str(tmp_path / f"out{i}"),
                                        "report": lambda p, m, i=i: notes[i].append(m)})
               for i, f in enumerate(folders)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == {tmp_path / "a": 1, tmp_path / "b": 1}
    assert overall[1] == 2  # a different folder doesn't wait
    assert sum(any(m.startswith("waiting for another download") for m in n) for n in notes) == 2



def test_list_clips_shows_each_clip_layout(tmp_path: Path):
    m = _manifest(tmp_path / "run")
    data = json.loads(m.read_text(encoding="utf-8"))
    data["render"] = {"layout": "fit"}
    data["clips"][1]["layout"] = "crop"
    m.write_text(json.dumps(data), encoding="utf-8")
    assert [c["layout"] for c in list_clips(str(m))["clips"]] == ["fit", "crop"]
    del data["render"]  # manifests from before render options were saved
    m.write_text(json.dumps(data), encoding="utf-8")
    assert list_clips(str(m))["clips"][0]["layout"] == "blur"


def test_rerender_clip_fills_in_the_current_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    m = _manifest(tmp_path / "run")
    calls: list[tuple] = []

    def fake(manifest: Path, n: int, start: float, end: float, log=print, layout=None) -> dict:
        calls.append((manifest, n, start, end, layout))
        log("re-rendering")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["clips"][n - 1].update(start=start, end=end, layout=layout or data["clips"][n - 1].get("layout"))
        manifest.write_text(json.dumps(data), encoding="utf-8")
        return data["clips"][n - 1]

    monkeypatch.setattr(pipeline, "rerender_clip", fake)
    out = rerender_clip(str(m.parent), 1, layout="crop")
    assert calls[-1] == (m, 1, 1.0, 21.5, "crop")
    assert out["n"] == 1 and out["layout"] == "crop" and out["log"] == ["re-rendering"]
    out = rerender_clip(str(m), 2, start=42.0)
    assert calls[-1] == (m, 2, 42.0, 60.0, None) and out["start"] == 42.0


def test_rerender_clip_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    m = _manifest(tmp_path / "run")
    monkeypatch.setattr(pipeline, "render", lambda *a, **k: pytest.fail("should not render"))
    with pytest.raises(ToolError, match="clip 3 doesn't exist, this run has 2"):
        rerender_clip(str(m), 3, layout="crop")
    with pytest.raises(ToolError, match="nothing to change"):
        rerender_clip(str(m), 1)
    with pytest.raises(ToolError, match="layout must be one of"):
        rerender_clip(str(m), 1, layout="zoom")
    with pytest.raises(ToolError, match="not found"):
        rerender_clip(str(tmp_path / "nope"), 1, layout="crop")


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_rerender_clip_changes_the_layout_for_real(tmp_path: Path):
    video = tmp_path / "talk.mp4"
    make_video(video, duration=40)
    out = make_clips(str(video), count=1, captions=False, layout="fit", out=str(tmp_path / "out"))
    clip = out["clips"][0]
    assert clip["layout"] == "fit"
    again = rerender_clip(out["manifest"], 1, end=clip["end"] - 2, layout="crop")
    assert again["layout"] == "crop" and again["end"] == pytest.approx(clip["end"] - 2)
    assert Path(again["file"]).is_file() and any("crop layout" in line for line in again["log"])


def test_rerender_reads_the_range_after_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import threading
    import time

    from clipwright import mcp_server

    m = _manifest(tmp_path / "run")
    calls: list[tuple[float, float]] = []
    monkeypatch.setattr(pipeline, "rerender_clip",
                        lambda manifest, n, start, end, log=print, layout=None: calls.append((start, end)))
    notes: list[str] = []
    mcp_server._RUN_LOCK.acquire()
    try:
        t = threading.Thread(target=rerender_clip, args=(str(m), 1),
                             kwargs={"layout": "crop", "report": lambda f, msg: notes.append(msg)})
        t.start()
        time.sleep(0.2)
        # the run being waited for writes new clips into the same folder
        data = json.loads(m.read_text(encoding="utf-8"))
        data["clips"][0].update(start=30.0, end=50.0)
        m.write_text(json.dumps(data), encoding="utf-8")
    finally:
        mcp_server._RUN_LOCK.release()
    t.join()
    assert notes == ["waiting for another clipwright call to finish"]
    assert calls == [(30.0, 50.0)]
