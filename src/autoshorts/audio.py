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
        capture_output=True, check=True,
    )
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


def silences(
    db: FloatArray, hop: float, threshold_db: float = -40.0, min_len: float = 0.4
) -> list[tuple[float, float]]:
    """Stretches quieter than threshold_db lasting at least min_len seconds, as (start, end)."""
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
