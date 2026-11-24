"""A cover image for each clip, taken from the rendered clip itself.

Taking it from the rendered 9:16 file (not the source) means it already has the layout, the
crop and the jump cuts. The frame is picked near the loudest moment, since that's usually why
the clip was chosen, but within a second or so of it the *stillest* frame wins: the frame with
the most motion is usually motion-blurred, which makes a bad cover.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import audio
from .motion import read_motion
from .probe import probe, require

HOP = 0.1          # loudness resolution in seconds
FPS = 10.0         # motion samples per second
NEAR = 1.0         # how far from the loudest moment the frame may be, in seconds
EDGE = 0.5         # stay away from the first/last half second (fades, cut-in frames)


def pick_time(loudness: npt.NDArray[np.float64] | None, energy: npt.NDArray[np.float32], duration: float,
              hop: float = HOP, fps: float = FPS) -> float:
    """Clip-relative time of the cover frame.

    loudness: dB per `hop` seconds (None without audio), energy: motion per frame at `fps`."""
    lo, hi = min(EDGE, duration / 2), max(duration - EDGE, duration / 2)
    peak = duration / 3  # no audio to go on: a third in is usually past any intro
    if loudness is not None and len(loudness):
        times = (np.arange(len(loudness)) + 0.5) * hop
        ok = (times >= lo) & (times <= hi)
        if ok.any():
            peak = float(times[ok][int(np.argmax(loudness[ok]))])
    if len(energy) < 2:
        return round(min(max(peak, lo), hi), 2)
    t = np.arange(len(energy)) / fps
    near = (np.abs(t - peak) <= NEAR) & (t >= lo) & (t <= hi)
    near[0] = False  # row 0 has no previous frame, its energy is a placeholder 0
    if not near.any():
        return round(min(max(peak, lo), hi), 2)
    candidates = np.flatnonzero(near)
    # least motion, ties broken by being closer to the peak
    best = min(candidates, key=lambda i: (float(energy[i]), abs(float(t[i]) - peak)))
    return round(float(t[best]), 2)


def grab(video: Path, t: float, out: Path, width: int = 720) -> None:
    """Writes the frame at t as a JPEG (720 wide by default: plenty for a cover, ~100 KB)."""
    proc = subprocess.run(
        [require("ffmpeg"), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-q:v", "3", str(out)],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0 or not out.is_file():
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"couldn't grab a frame from {video.name}:\n{tail}")


def make_thumbnail(clip: Path, out: Path) -> float:
    """Picks and writes the cover for a rendered clip. Returns the chosen time."""
    info = probe(clip)
    loudness = None
    if info.has_audio:
        loudness = audio.loudness_curve(audio.decode_pcm(clip), hop=HOP)
    motion = read_motion(clip, info.video_stream, info.width, info.height, fps=FPS)
    t = pick_time(loudness, motion.energy, info.duration)
    grab(clip, t, out)
    return t
