"""Speech to text with word timestamps (faster-whisper, optional)."""

from __future__ import annotations

import functools
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

# whisper takes raw samples at 16 kHz, which is what audio.decode_pcm produces
from .audio import SAMPLE_RATE


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _add_cuda_dll_dirs() -> None:
    """The `gpu` extra installs cuBLAS/cuDNN as pip wheels (nvidia-*). On Windows their DLLs
    live in site-packages/nvidia/<lib>/bin, which isn't on the DLL search path by default."""
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _is_cuda_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", "libcudart"))


Fraction = Callable[[float], None]


def _run(audio: npt.NDArray[np.float32], model_size: str, device: str, language: str | None,
         on_progress: Fraction | None = None) -> list[Word]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True, language=language)
    words: list[Word] = []
    duration = len(audio) / SAMPLE_RATE
    # segments is a lazy generator, the actual GPU work happens while iterating
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(float(w.start), float(w.end), text))
        if on_progress and duration > 0:
            on_progress(float(seg.end) / duration)
    return words


def monotonic(cb: Fraction | None) -> Fraction | None:
    """Clamps to [0, 1] and never reports a smaller value than before (the CPU retry after a
    CUDA failure starts over from 0)."""
    if cb is None:
        return None
    best = 0.0

    def report(f: float) -> None:
        nonlocal best
        f = min(1.0, max(0.0, f))
        if f > best:
            best = f
            cb(f)

    return report


def transcribe(
    audio: npt.NDArray[np.float32],
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
    log: Callable[[str], None] = print,
    on_progress: Fraction | None = None,
) -> list[Word]:
    """audio: 16 kHz mono float32, the same buffer the loudness analysis already decoded.

    Passing samples instead of a path skips faster-whisper's own PyAV decoding (a second full
    decode, and it breaks with some PyAV versions).

    device="auto" tries CUDA first and falls back to CPU if the CUDA libraries are missing,
    which faster-whisper only reports once decoding starts.

    on_progress gets the fraction of the audio transcribed so far, after each whisper segment.
    """
    progress = monotonic(on_progress)
    if device == "cpu":
        return _run(audio, model_size, "cpu", language, progress)
    _add_cuda_dll_dirs()
    try:
        return _run(audio, model_size, "cuda", language, progress)
    except (RuntimeError, ValueError, OSError) as err:
        if device == "cuda" or not _is_cuda_error(err):
            raise
        log(f"GPU transcription unavailable ({err}), using CPU. `pip install autoshorts[gpu]` fixes this.")
        return _run(audio, model_size, "cpu", language, progress)


def save_words(words: list[Word], path: Path) -> None:
    """[[start, end, text], ...] so captions can be rebuilt later without running whisper again."""
    path.write_text(json.dumps([[w.start, w.end, w.text] for w in words], ensure_ascii=False), encoding="utf-8")


def load_words(path: Path) -> list[Word]:
    return [Word(float(a), float(b), str(t)) for a, b, t in json.loads(path.read_text(encoding="utf-8"))]


def speech_density(words: list[Word], duration: float, hop: float) -> npt.NDArray[np.float64]:
    """Words per second on the same timeline as the loudness curve."""
    n = int(duration / hop)
    out = np.zeros(n)
    for w in words:
        i = int(((w.start + w.end) / 2) / hop)
        if 0 <= i < n:
            out[i] += 1.0 / hop
    return out


def sentence_ends(words: list[Word]) -> list[float]:
    """Times right after a word that ends a sentence, or before a long pause."""
    ends: list[float] = []
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w.text.endswith((".", "!", "?")) or (nxt is not None and nxt.start - w.end > 0.7):
            ends.append(round(w.end + 0.15, 2))
    return ends
