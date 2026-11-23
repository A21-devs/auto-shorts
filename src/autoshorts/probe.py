"""Thin wrappers around ffprobe / ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        [require("ffprobe"), "-v", "error", "-show_entries",
         "format=duration:stream=index,codec_type,width,height,avg_frame_rate,disposition",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    # OBS / Roblox recordings sometimes have a cover-art stream, skip attached pictures
    videos = [s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError(f"{path.name} has no video stream")
    v = max(videos, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
    return MediaInfo(
        path=path,
        duration=float(data["format"]["duration"]),
        width=int(v["width"]),
        height=int(v["height"]),
        fps=_fps(v.get("avg_frame_rate", "30/1")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_stream=int(v["index"]),
    )
