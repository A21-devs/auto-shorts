"""MCP server so AI agents (Claude Code, Claude Desktop, ...) can make clips.

Run with `clipwright mcp` (stdio). Needs the optional extra: `pip install "clipwright[mcp]"`.

The tool functions below are plain Python and don't import the MCP SDK, so they're testable
without it; `build_server` wraps them. Validation is the review page's `parse_job`, so an agent
gets the same limits and error messages as a person using `serve`.

stdout is the protocol channel on stdio, so nothing here may print to it: the pipeline's log
goes to stderr and is also returned to the agent with the result.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import download, pipeline
from .server import BadRequest, parse_job

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

LOG_TAIL = 40  # log lines returned with a result; the full log still goes to stderr


class ToolError(ValueError):
    """A bad request from the agent, reported back as the tool's error message."""


def _stderr_log(lines: list[str]) -> pipeline.Log:
    def log(msg: str) -> None:
        lines.append(msg)
        print(msg, file=sys.stderr, flush=True)

    return log


def make_clips(video: str, count: int = 6, platform: str | None = None, layout: str = "blur",
               captions: bool = True, jumpcut: bool = False, model: str = "small",
               out: str | None = None) -> dict[str, Any]:
    """Cuts a local video into vertical shorts and returns where they are plus each clip's range,
    score and files. Blocks until the run is done (minutes for a long video)."""
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
    manifest = pipeline.run(job, log=_stderr_log(lines))
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

    return wrapper


def build_server() -> MCPServer:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        raise RuntimeError('the MCP server needs the MCP SDK: pip install "clipwright[mcp]"') from None
    server = MCPServer("clipwright", instructions="Cuts long local videos into vertical 9:16 shorts with captions.")
    server.tool()(_reported(make_clips))
    server.tool()(_reported(list_clips))
    return server


def main() -> int:
    try:
        server = build_server()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    server.run()  # stdio
    return 0
