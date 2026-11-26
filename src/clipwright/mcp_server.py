"""MCP server so AI agents (Claude Code, Claude Desktop, ...) can make clips.

Run with `clipwright mcp` (stdio). Needs the optional extra: `pip install "clipwright[mcp]"`.

The tool functions below are plain Python and don't import the MCP SDK, so they're testable
without it; `build_server` wraps them. Validation is the review page's `parse_job`, so an agent
gets the same limits and error messages as a person using `serve`.

stdout is the protocol channel on stdio, so nothing here may print to it: the pipeline's log
goes to stderr and is also returned to the agent with the result.

`make_clips` sends progress notifications (when the client asks for them with a progress token),
so a long run, or one queued behind another, doesn't look hung and hit the client's timeout.
"""

from __future__ import annotations

import functools
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import download, pipeline
from .server import BadRequest, parse_job

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

LOG_TAIL = 40  # log lines returned with a result; the full log still goes to stderr
# The SDK runs each sync tool call on its own worker thread, so parallel calls would run parallel
# pipelines. Queue them instead, like serve's single worker: two Whisper models and two encodes
# on one GPU are slower than doing them in order.
_RUN_LOCK = threading.Lock()


# Where each pipeline stage sits in the 0..1 progress sent to the agent. Rough shares of a run's
# time on a long video with captions; without captions "transcribe" is skipped and progress jumps.
STAGE_SPAN = {"audio": (0.0, 0.05), "scenes": (0.05, 0.25), "transcribe": (0.25, 0.55),
              "render": (0.55, 0.99), "done": (1.0, 1.0)}
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


def stage_progress(report: Report, after: float = -1.0) -> pipeline.Progress:
    """Turns the pipeline's progress(stage, fraction) into one rising fraction for the whole run.
    Only sends a step of at least PROGRESS_STEP (and always the end), never a step back. MCP wants
    every notification above the last one, so `after` is what was already sent for this call."""
    last = [after]

    def progress(stage: str, fraction: float) -> None:
        span = STAGE_SPAN.get(stage)
        if span is None:
            return
        lo, hi = span
        overall = lo + (hi - lo) * min(max(fraction, 0.0), 1.0)
        if overall < last[0] + PROGRESS_STEP and not (overall >= 1.0 > last[0]):
            return
        last[0] = overall
        report(overall, stage if stage == "done" else f"{stage} {fraction:.0%}")

    return progress


def make_clips(video: str, count: int = 6, platform: str | None = None, layout: str = "blur",
               captions: bool = True, jumpcut: bool = False, model: str = "small",
               out: str | None = None, report: Report | None = None) -> dict[str, Any]:
    """Cuts a local video into vertical shorts and returns where they are plus each clip's range,
    score and files. Blocks until the run is done (minutes for a long video) and reports progress
    while it runs."""
    if download.is_url(video.strip().strip('"')):
        raise ToolError("links aren't supported here yet, download the video and pass its path")
    body: dict[str, Any] = {"video": video, "count": count, "layout": layout, "captions": captions,
                            "jumpcut": jumpcut, "model": model}
    if platform is not None:
        body["platform"] = platform
    if out is not None:
        body["out"] = out
    try:
        job = parse_job(body)
    except BadRequest as e:
        raise ToolError(str(e)) from None
    lines: list[str] = []
    send = report if report is not None else lambda _f, _m: None
    sent = -1.0
    if not _RUN_LOCK.acquire(blocking=False):
        sent = 0.0
        send(sent, "waiting for another make_clips call to finish")
        _RUN_LOCK.acquire()
    progress = stage_progress(send, after=sent)
    try:
        manifest = pipeline.run(job, log=_stderr_log(lines), progress=progress)
    finally:
        _RUN_LOCK.release()
    result = list_clips(str(manifest))
    result["log"] = lines[-LOG_TAIL:]
    return result


def list_clips(manifest: str) -> dict[str, Any]:
    """Clips of a finished run, from its manifest.json (or the folder that holds it)."""
    path = Path(manifest.strip().strip('"')).expanduser()
    if path.is_dir():
        path = path / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        clips = data["clips"]
    except FileNotFoundError:
        raise ToolError(f"{path} not found") from None
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ToolError(f"{path} isn't a clipwright manifest ({e})") from None
    folder = path.parent.resolve()
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
            "thumbnail": str(folder / c["thumbnail"]) if c.get("thumbnail") else None,
            "srt": str(folder / c["srt"]) if c.get("srt") else None,
            "copy": c.get("copy"),
            "keep": c.get("keep"),
        })
    return {"manifest": str(path.resolve()), "source": data.get("source"), "platform": data.get("platform"),
            "clips": out}


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
    return server


def _make_clips_tool() -> Callable[..., dict[str, Any]]:
    """make_clips with the SDK's request Context, whose progress goes out as notifications.

    The SDK runs sync tools on a worker thread (anyio.to_thread), so the async report_progress is
    sent through anyio.from_thread back on the event loop. The SDK finds the context parameter
    by its annotation, which has to resolve at runtime, hence the import and the explicit
    __annotations__ (this module's annotations are strings and Context is a TYPE_CHECKING import).
    """
    import anyio.from_thread
    from mcp.server.mcpserver import Context

    def tool(video: str, count: int = 6, platform: str | None = None, layout: str = "blur",
             captions: bool = True, jumpcut: bool = False, model: str = "small",
             out: str | None = None, ctx: Context | None = None) -> dict[str, Any]:
        report: Report | None = None
        if ctx is not None and _in_request(ctx):
            def report(fraction: float, message: str) -> None:
                anyio.from_thread.run(ctx.report_progress, fraction, 1.0, message)
        return make_clips(video, count=count, platform=platform, layout=layout, captions=captions,
                          jumpcut=jumpcut, model=model, out=out, report=report)

    tool.__annotations__["ctx"] = Context | None
    return tool


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
