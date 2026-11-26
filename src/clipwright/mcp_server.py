"""MCP server so AI agents (Claude Code, Claude Desktop, ...) can make clips.

Run with `clipwright mcp` (stdio). Needs the optional extra: `pip install "clipwright[mcp]"`.

The tool functions below are plain Python and don't import the MCP SDK, so they're testable
without it; `build_server` wraps them. Validation is the review page's `parse_job`, so an agent
gets the same limits and error messages as a person using `serve`.

stdout is the protocol channel on stdio, so nothing here may print to it: the pipeline's log
goes to stderr and is also returned to the agent with the result.

`make_clips` sends progress notifications (when the client asks for them with a progress token),
so a long run, or one queued behind another, doesn't look hung and hit the client's timeout.
It also takes links (needs `clipwright[link]`), downloaded like `serve` does.

`rerender_clip` redoes one clip of a finished run with a new range and/or layout. It shares the
run lock with `make_clips`, so it waits for a run in progress instead of encoding beside it.
"""

from __future__ import annotations

import functools
import json
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import download, pipeline
from .server import DEFAULT_DOWNLOADS, BadRequest, parse_job
from .subtitles import sidecar

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

LOG_TAIL = 40  # log lines returned with a result; the full log still goes to stderr
# The SDK runs each sync tool call on its own worker thread, so parallel calls would run parallel
# pipelines. Queue them instead, like serve's single worker: two Whisper models and two encodes
# on one GPU are slower than doing them in order.
_RUN_LOCK = threading.Lock()
# Downloads run outside _RUN_LOCK, but two into one folder can be the same video (an agent retrying,
# or youtu.be/x and youtube.com/watch?v=x), and yt-dlp would write the same .part file twice.
# One at a time per folder; the second then finds the finished file and skips the download.
_DOWNLOAD_LOCKS: dict[Path, threading.Lock] = {}
_DOWNLOAD_LOCKS_GUARD = threading.Lock()


# Where each pipeline stage sits in the 0..1 progress sent to the agent. Rough shares of a run's
# time on a long video with captions; without captions "transcribe" is skipped and progress jumps.
STAGE_SPAN = {"audio": (0.0, 0.05), "scenes": (0.05, 0.25), "transcribe": (0.25, 0.55),
              "render": (0.55, 0.99), "done": (1.0, 1.0)}
# A link spends a good part of the run downloading (a VOD is gigabytes), the rest is scaled after it.
DOWNLOAD_SHARE = 0.3
LINK_SPAN = {"download": (0.0, DOWNLOAD_SHARE),
             **{k: (DOWNLOAD_SHARE + (1 - DOWNLOAD_SHARE) * lo, DOWNLOAD_SHARE + (1 - DOWNLOAD_SHARE) * hi)
                for k, (lo, hi) in STAGE_SPAN.items()}}
PROGRESS_STEP = 0.01  # whisper reports every segment; don't send a notification for each one

# progress(fraction of the whole run, message)
Report = Callable[[float, str], None]


class ToolError(ValueError):
    """A bad request from the agent, reported back as the tool's error message."""


def _stderr_log(lines: list[str]) -> pipeline.Log:
    def log(msg: str) -> None:
        lines.append(msg)
        print(msg, file=sys.stderr, flush=True)

    return log


class RunProgress:
    """Turns progress(stage, fraction) from the download and the pipeline into one rising fraction
    for the whole call. Only sends a step of at least PROGRESS_STEP (and always the end), never a
    step back: MCP wants every notification's progress above the one before."""

    def __init__(self, report: Report, spans: dict[str, tuple[float, float]] = STAGE_SPAN) -> None:
        self.report, self.spans, self.last = report, spans, -1.0

    def __call__(self, stage: str, fraction: float) -> None:
        span = self.spans.get(stage)
        if span is None:
            return
        lo, hi = span
        overall = lo + (hi - lo) * min(max(fraction, 0.0), 1.0)
        if overall < self.last + PROGRESS_STEP and not (overall >= 1.0 > self.last):
            return
        self.last = overall
        self.report(overall, stage if stage == "done" else f"{stage} {fraction:.0%}")

    def note(self, message: str) -> None:
        """A message between steps, sent just above the last value so it still counts as progress."""
        self.last = max(self.last + PROGRESS_STEP / 10, 0.0)
        self.report(self.last, message)


def make_clips(video: str, count: int = 6, platform: str | None = None, layout: str = "blur",
               captions: bool = True, jumpcut: bool = False, model: str = "small", aspect: str | None = None,
               out: str | None = None, downloads: str | None = None,
               report: Report | None = None) -> dict[str, Any]:
    """Cuts a video into vertical shorts and returns where they are plus each clip's range,
    score and files. `video` is a local file or a link (YouTube, Twitch, ...), which is downloaded
    into `downloads` (default: Videos/clipwright) with its subtitles first. Blocks until the run is
    done (minutes for a long video) and reports progress while it runs."""
    body: dict[str, Any] = {"video": video, "count": count, "layout": layout, "captions": captions,
                            "jumpcut": jumpcut, "model": model}
    if platform is not None:
        body["platform"] = platform
    if aspect is not None:
        body["aspect"] = aspect
    if out is not None:
        body["out"] = out
    try:
        job = parse_job(body)
    except BadRequest as e:
        raise ToolError(str(e)) from None
    lines: list[str] = []
    log = _stderr_log(lines)
    progress = RunProgress(report if report is not None else lambda _f, _m: None,
                           LINK_SPAN if job.url is not None else STAGE_SPAN)
    if job.url is not None:
        # outside the lock: a download doesn't fight over the GPU, a queued call can fetch meanwhile
        job = _fetch(job, Path(downloads).expanduser() if downloads else DEFAULT_DOWNLOADS, log, progress)
    if not _RUN_LOCK.acquire(blocking=False):
        progress.note("waiting for another make_clips call to finish")
        _RUN_LOCK.acquire()
    try:
        manifest = pipeline.run(job, log=log, progress=progress)
    finally:
        _RUN_LOCK.release()
    result = list_clips(str(manifest))
    result["log"] = lines[-LOG_TAIL:]
    return result


def _fetch(job: pipeline.Job, folder: Path, log: pipeline.Log, progress: RunProgress) -> pipeline.Job:
    """Downloads a link job's video (and subtitles when captions are on, like the CLI and serve)
    and points the job at the file."""
    assert job.url is not None
    lang = (job.language or "en") if job.captions else None
    with _DOWNLOAD_LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.setdefault(folder.resolve(), threading.Lock())
    if not lock.acquire(blocking=False):
        progress.note(f"waiting for another download into {folder}")
        lock.acquire()
    try:
        path = download.download(job.url, folder, log=log, on_progress=lambda f: progress("download", f),
                                 subtitles=lang)
    finally:
        lock.release()
    progress("download", 1.0)
    return replace(job, source=path, out_dir=job.out_dir or path.with_name(f"{path.stem}_shorts"),
                   srt=sidecar(path) if job.captions else None)


def _manifest_path(manifest: str) -> Path:
    path = Path(manifest.strip().strip('"')).expanduser()
    return path / "manifest.json" if path.is_dir() else path


def list_clips(manifest: str) -> dict[str, Any]:
    """Clips of a finished run, from its manifest.json (or the folder that holds it)."""
    path = _manifest_path(manifest)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        clips = data["clips"]
    except FileNotFoundError:
        raise ToolError(f"{path} not found") from None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ToolError(f"{path} isn't a clipwright manifest ({e})") from None
    folder = path.parent.resolve()
    run_layout = (data.get("render") or {}).get("layout", "blur")
    out = []
    for n, c in enumerate(clips, 1):
        out.append({
            "n": n,
            "file": str(folder / c["file"]),
            "start": c.get("start"),
            "end": c.get("end"),
            "length": c.get("length"),
            "score": c.get("score"),
            "reasons": c.get("reasons", []),
            "layout": c.get("layout") or run_layout,
            "thumbnail": str(folder / c["thumbnail"]) if c.get("thumbnail") else None,
            "srt": str(folder / c["srt"]) if c.get("srt") else None,
            "copy": c.get("copy"),
            "keep": c.get("keep"),
        })
    return {"manifest": str(path.resolve()), "source": data.get("source"), "platform": data.get("platform"),
            "clips": out}


def _clip(path: Path, n: int) -> dict[str, Any]:
    clips: list[dict[str, Any]] = list_clips(str(path))["clips"]
    if not 1 <= n <= len(clips):
        raise ToolError(f"clip {n} doesn't exist, this run has {len(clips)}")
    return clips[n - 1]


def rerender_clip(manifest: str, n: int, start: float | None = None, end: float | None = None,
                  layout: str | None = None, report: Report | None = None) -> dict[str, Any]:
    """Re-renders clip n (1-based, as in list_clips) of a finished run and returns it updated.
    start/end are seconds in the source video and default to the clip's current range. layout
    (blur, crop, follow, speaker, fit) changes this clip only and is kept for later re-renders."""
    path = _manifest_path(manifest)
    if start is None and end is None and layout is None:
        raise ToolError("nothing to change: give start/end, layout, or both")
    _clip(path, n)  # fail fast on a bad manifest or n, before waiting for the lock
    lines: list[str] = []
    progress = RunProgress(report if report is not None else lambda _f, _m: None)
    if not _RUN_LOCK.acquire(blocking=False):
        progress.note("waiting for another clipwright call to finish")
        _RUN_LOCK.acquire()
    try:
        # read again under the lock: the run we waited for may have replaced this folder's clips
        clip = _clip(path, n)
        pipeline.rerender_clip(path, n, clip["start"] if start is None else start,
                               clip["end"] if end is None else end, log=_stderr_log(lines), layout=layout)
        result = _clip(path, n)
    except ValueError as e:  # bad range or layout: the agent's to fix, the old clip is untouched
        raise ToolError(str(e)) from None
    finally:
        _RUN_LOCK.release()
    result["log"] = lines[-LOG_TAIL:]
    return result


def _reported(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """The SDK only shows the agent the message of its own ToolError; anything else becomes
    "Error executing tool make_clips". A missing ffmpeg or a failed render is something the
    agent (and the person behind it) needs to read, so every failure is passed on as one."""
    from mcp.server.mcpserver.exceptions import ToolError as SdkToolError

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise SdkToolError(str(e) or type(e).__name__) from e

    # the SDK finds the Context parameter from get_type_hints(wrapper); on 3.14 functools.wraps
    # copies __annotate__ instead of __annotations__, which would leave it the wrapper's own
    wrapper.__annotations__ = dict(fn.__annotations__)
    return wrapper


def build_server() -> MCPServer:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        raise RuntimeError('the MCP server needs the MCP SDK: pip install "clipwright[mcp]"') from None
    server = MCPServer("clipwright", instructions="Cuts long local videos into vertical 9:16 shorts with captions.")
    server.tool(name="make_clips", description=make_clips.__doc__)(_reported(_make_clips_tool()))
    server.tool()(_reported(list_clips))
    server.tool(name="rerender_clip", description=rerender_clip.__doc__)(_reported(_rerender_clip_tool()))
    return server


def _make_clips_tool() -> Callable[..., dict[str, Any]]:
    """make_clips with the SDK's request Context, whose progress goes out as notifications.

    The SDK runs sync tools on a worker thread (anyio.to_thread), so the async report_progress is
    sent through anyio.from_thread back on the event loop. The SDK finds the context parameter
    by its annotation, which has to resolve at runtime, hence the import and the explicit
    __annotations__ (this module's annotations are strings and Context is a TYPE_CHECKING import).
    """
    from mcp.server.mcpserver import Context

    def tool(video: str, count: int = 6, platform: str | None = None, layout: str = "blur",
             captions: bool = True, jumpcut: bool = False, model: str = "small",
             out: str | None = None, downloads: str | None = None, aspect: str | None = None,
             ctx: Context | None = None) -> dict[str, Any]:
        return make_clips(video, count=count, platform=platform, layout=layout, captions=captions,
                          jumpcut=jumpcut, model=model, out=out, downloads=downloads, aspect=aspect,
                          report=_reporter(ctx))

    tool.__annotations__["ctx"] = Context | None
    return tool


def _rerender_clip_tool() -> Callable[..., dict[str, Any]]:
    """rerender_clip with the request Context, see _make_clips_tool."""
    from mcp.server.mcpserver import Context

    def tool(manifest: str, n: int, start: float | None = None, end: float | None = None,
             layout: str | None = None, ctx: Context | None = None) -> dict[str, Any]:
        return rerender_clip(manifest, n, start=start, end=end, layout=layout, report=_reporter(ctx))

    tool.__annotations__["ctx"] = Context | None
    return tool


def _reporter(ctx: Any) -> Report | None:
    """Sends progress through ctx from the tool's worker thread, or None outside a client request."""
    if ctx is None or not _in_request(ctx):
        return None
    import anyio.from_thread

    def report(fraction: float, message: str) -> None:
        anyio.from_thread.run(ctx.report_progress, fraction, 1.0, message)

    return report


def _in_request(ctx: Any) -> bool:
    """False when the tool is called directly (server.call_tool in tests), not from a client."""
    try:
        _ = ctx.request_context
    except ValueError:
        return False
    return True


def main() -> int:
    try:
        server = build_server()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    server.run()  # stdio
    return 0
