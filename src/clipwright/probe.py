"""Thin wrappers around ffprobe / ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolMissing(RuntimeError):
    pass


def require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise ToolMissing(f"{tool} was not found on PATH. Install ffmpeg (it includes ffprobe) and try again.")
    return path


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_stream: int  # absolute stream index, for -map 0:<n>


def _fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 30.0


_IMAGE_CODECS = {"mjpeg", "png", "bmp", "gif", "webp", "tiff"}


def _video_rank(s: dict[str, Any]) -> tuple[int, int]:
    """Higher is better. Real video beats thumbnails, then bigger beats smaller.

    Roblox and some phone recordings store a 1-frame JPEG thumbnail as a normal video stream
    (no attached_pic flag) that's bigger than the actual video, so resolution alone picks it.
    """
    is_picture = (
        bool(s.get("disposition", {}).get("attached_pic"))
        or s.get("codec_name") in _IMAGE_CODECS
        # exactly one frame = still image. "0" just means ffprobe doesn't know (fragmented MP4).
        or str(s.get("nb_frames", "")) == "1"
    )
    return (0 if is_picture else 1, int(s.get("width", 0)) * int(s.get("height", 0)))


def probe(path: Path) -> MediaInfo:
    # absolute from here on: render runs ffmpeg inside the output folder, and the manifest keeps
    # this path for re-renders started later from another working directory
    path = path.resolve()
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-show_entries",
         "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,disposition",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if not videos:
        raise ValueError(f"{path.name} has no video stream")
    v = max(videos, key=_video_rank)
    return MediaInfo(
        path=path,
        duration=float(data["format"]["duration"]),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=_fps(v.get("avg_frame_rate", "30/1")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_stream=int(v["index"]),
    )
