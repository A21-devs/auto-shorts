"""A cover image for each clip.

The frame is picked on the rendered clip (its loudness and motion, jump cuts already applied),
near the loudest moment, since that's usually why the clip was chosen. Within a second or so of
it the *stillest* frame wins: the frame with the most motion is usually motion-blurred, which
makes a bad cover.

The picture itself is rendered from the source with the clip's layout and crop but without the
captions (render.render_cover), otherwise covers carry half a caption line and look like a paused
video. `make_thumbnail` grabs straight from the clip, for clips that have no source to go back to.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import audio, jumpcut
from .motion import read_motion
from .probe import MediaInfo, probe, require
from .render import RenderOptions, render_cover

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


def clip_time(clip: Path) -> float:
    """Time in the rendered clip (0 = its first frame) that makes the best cover."""
    info = probe(clip)
    loudness = None
    if info.has_audio:
        loudness = audio.loudness_curve(audio.decode_pcm(clip), hop=HOP)
    motion = read_motion(clip, info.video_stream, info.width, info.height, fps=FPS)
    return pick_time(loudness, motion.energy, info.duration)


def make_thumbnail(clip: Path, out: Path) -> float:
    """Picks and writes the cover for a rendered clip, frame grabbed from the clip itself (so
    with its captions). Returns the chosen time."""
    t = clip_time(clip)
    grab(clip, t, out)
    return t


def make_cover(clip: Path, source: MediaInfo, start: float, out: Path, opts: RenderOptions,
               keep: jumpcut.Ranges | None = None, crop_x: str | None = None) -> float:
    """Picks the cover time on the rendered clip and renders that frame from the source without
    captions. `start` is where the clip starts in the source, `keep` the jump-cut ranges it was
    rendered with and `crop_x` the follow/speaker window (render() returns it). Returns the
    clip time used."""
    t = clip_time(clip)
    at = start + (jumpcut.source_time(t, keep) if keep else t)
    render_cover(source, at, out, opts, crop_x, clip_time=t)
    return t
