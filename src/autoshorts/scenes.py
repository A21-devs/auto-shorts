"""Scene cut detection with ffmpeg's scene score.

Runs on a tiny low-fps copy of the video so a 15 minute recording takes seconds.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .probe import require

_PTS = re.compile(r"pts_time:([0-9.]+)")


def parse_showinfo(stderr: str) -> list[float]:
    """Pulls frame timestamps out of ffmpeg showinfo output."""
    return [float(m.group(1)) for line in stderr.splitlines() if "Parsed_showinfo" in line
            for m in [_PTS.search(line)] if m]


def detect_cuts(path: Path, video_stream: int = 0, threshold: float = 0.35, fps: int = 5) -> list[float]:
    vf = f"fps={fps},scale=192:-2,select='gt(scene,{threshold})',showinfo"
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-map", f"0:{video_stream}",
         "-an", "-vf", vf, "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"scene detection failed for {path.name}:\n{tail}")
    return parse_showinfo(proc.stderr)
