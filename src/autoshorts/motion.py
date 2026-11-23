"""Where things move in the frame.

Decodes a tiny grayscale copy of the video (a few fps, about 96 px wide) through an ffmpeg pipe
and measures how much each column changes from one frame to the next. That says how much is
happening (the mean over columns) and where it's happening (which columns light up).

Frames are read one at a time and only the per-column differences are kept, so memory stays at
`frames x 96` floats even for a 2 hour video.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .probe import require

FloatArray = npt.NDArray[np.float32]

STILL = 1.0  # mean abs difference (0-255) below which a window counts as "nothing moving"


@dataclass(frozen=True)
class Motion:
    fps: float
    # (frames, width): mean absolute difference from the previous frame, per column, 0-255.
    # Row i is frame i at t = i / fps. Row 0 is all zeros (nothing to compare it with).
    columns: FloatArray

    @property
    def energy(self) -> FloatArray:
        """How much changes in each frame, averaged over the whole picture."""
        if self.columns.size == 0:
            return np.zeros(0, dtype=np.float32)
        return self.columns.mean(axis=1)

    @property
    def duration(self) -> float:
        return len(self.columns) / self.fps


def small_size(width: int, height: int, target: int = 96) -> tuple[int, int]:
    """Downscaled size with the source aspect ratio, height rounded to an even number."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad video size {width}x{height}")
    return target, max(2, round(target * height / width / 2) * 2)


def read_motion(path: Path, video_stream: int, width: int, height: int, start: float | None = None,
                duration: float | None = None, fps: float = 5.0, size: int = 96) -> Motion:
    """Motion of [start, start + duration) (the whole video by default) sampled at `fps`."""
    w, h = small_size(width, height, size)
    cmd = [require("ffmpeg"), "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-map", f"0:{video_stream}", "-an",
            "-vf", f"fps={fps:g},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"]
    frame_bytes = w * h
    rows: list[FloatArray] = [np.zeros(w, dtype=np.float32)]
    # stderr goes to a file: a broken input can print more than a pipe buffer holds, and nothing
    # reads stderr until stdout is done, which would deadlock
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        prev: npt.NDArray[np.int16] | None = None
        finished = False
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                cur = np.frombuffer(buf, dtype=np.uint8).reshape(h, w).astype(np.int16)
                if prev is not None:
                    rows.append(np.abs(cur - prev).mean(axis=0).astype(np.float32))
                prev = cur
            finished = True
        finally:
            # an exception mid-read (Ctrl+C, a failed job in `serve`) must not leave ffmpeg
            # decoding the rest of the video in the background
            proc.stdout.close()
            if not finished:
                proc.kill()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            tail = "\n".join(err.read().decode(errors="replace").strip().splitlines()[-5:])
            raise RuntimeError(f"motion analysis failed for {path.name}:\n{tail}")
    if prev is None:  # no frames at all (e.g. start past the end)
        return Motion(fps, np.zeros((0, w), dtype=np.float32))
    return Motion(fps, np.vstack(rows))


def focus(motion: Motion, window: float = 1.0, still: float = STILL) -> FloatArray:
    """Horizontal position of the action per `window` seconds, 0 = left edge, 1 = right edge.

    The per-frame median column is subtracted first, so motion spread over the whole picture
    (a camera pan, a fade) cancels out and only motion concentrated in part of the frame pulls
    the centre. Windows where nothing stands out are NaN, meaning "no opinion, stay put".
    """
    cols = motion.columns
    per = max(1, int(round(window * motion.fps)))
    n = len(cols) // per + (1 if len(cols) % per else 0)
    out = np.full(n, np.nan, dtype=np.float32)
    if cols.size == 0:
        return out
    excess = np.clip(cols - np.median(cols, axis=1, keepdims=True), 0, None)
    x = (np.arange(cols.shape[1], dtype=np.float32) + 0.5) / cols.shape[1]
    for i in range(n):
        # row 0 is the placeholder (no previous frame), it would drag the first window down
        lo, hi = max(1, i * per), (i + 1) * per
        if lo >= min(hi, len(cols)) or float(cols[lo:hi].mean()) < still:
            continue
        weights = excess[lo:hi].sum(axis=0)
        total = float(weights.sum())
        if total > 0:
            out[i] = float((weights * x).sum() / total)
    return out
