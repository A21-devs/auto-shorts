"""Audio loudness analysis.

Decodes the audio track to 16 kHz mono PCM through ffmpeg (no temp files) and turns it
into a loudness curve: one value per `hop` seconds, in dBFS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .probe import require

SAMPLE_RATE = 16_000
SILENCE_DB = -50.0

FloatArray = npt.NDArray[np.float64]


def decode_pcm(path: Path, sample_rate: int = SAMPLE_RATE) -> npt.NDArray[np.float32]:
    """Returns the first audio track as float32 mono in [-1, 1]."""
    proc = subprocess.run(
        [require("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.decode(errors="replace").strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg could not read the audio of {path.name}:\n{tail}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0)


def loudness_curve(samples: npt.NDArray[np.float32], sample_rate: int = SAMPLE_RATE, hop: float = 0.5) -> FloatArray:
    """RMS loudness per window in dBFS. Silent windows are clamped to SILENCE_DB."""
    win = max(1, int(sample_rate * hop))
    n = len(samples) // win
    if n == 0:
        return np.zeros(0)
    frames = samples[: n * win].reshape(n, win).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(db, SILENCE_DB)


def zscore(x: FloatArray) -> FloatArray:
    """Standardize so loud/quiet is relative to this video, not absolute levels."""
    if len(x) == 0:
        return x
    std = float(np.std(x))
    return (x - float(np.mean(x))) / std if std > 1e-9 else np.zeros_like(x)


def quiet_threshold(db: FloatArray, floor_db: float = -40.0, percentile: float = 10.0) -> float:
    """What counts as "quiet" for this video.

    Real silence (-40 dB) when the video has it. Gameplay with constant music never gets that
    quiet, so fall back to its own quietest 10% so there are still cut points.
    """
    if len(db) == 0 or bool(np.any(db < floor_db)):
        return floor_db
    return float(np.percentile(db, percentile))


def silences(
    db: FloatArray, hop: float, threshold_db: float | None = None, min_len: float = 0.4
) -> list[tuple[float, float]]:
    """Stretches quieter than threshold_db (default: quiet_threshold) lasting min_len+ seconds, as (start, end)."""
    if threshold_db is None:
        threshold_db = quiet_threshold(db)
    out: list[tuple[float, float]] = []
    start: int | None = None
    for i, v in enumerate(db):
        if v < threshold_db:
            if start is None:
                start = i
        elif start is not None:
            if (i - start) * hop >= min_len:
                out.append((start * hop, i * hop))
            start = None
    if start is not None and (len(db) - start) * hop >= min_len:
        out.append((start * hop, len(db) * hop))
    return out
