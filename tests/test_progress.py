import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

from autoshorts import transcribe
from autoshorts.transcribe import monotonic


@dataclass
class FakeWord:
    start: float
    end: float
    word: str


@dataclass
class FakeSegment:
    end: float
    words: list[FakeWord] = field(default_factory=list)


def install_fake_whisper(monkeypatch: pytest.MonkeyPatch, cuda_fails_after: int | None = None) -> None:
    """faster_whisper stand-in: 4 segments ending at 2.5/5/7.5/10s. On cuda it can blow up mid-way."""

    class WhisperModel:
        def __init__(self, _size: str, device: str, compute_type: str) -> None:
            self.device = device

        def transcribe(self, _audio, **_kw):
            def gen():
                for i in range(1, 5):
                    if self.device == "cuda" and cuda_fails_after is not None and i > cuda_fails_after:
                        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                    yield FakeSegment(i * 2.5, [FakeWord(i * 2.5 - 1, i * 2.5 - 0.5, f"w{i}")])
            return gen(), None

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    monkeypatch.setattr(transcribe, "_add_cuda_dll_dirs", lambda: None)


TEN_SECONDS = np.zeros(10 * transcribe.SAMPLE_RATE, dtype=np.float32)


def test_monotonic_clamps_and_never_goes_back():
    seen: list[float] = []
    report = monotonic(seen.append)
    assert report is not None
    for f in [0.2, 0.1, 0.5, 0.5, 1.7, 0.9]:
        report(f)
    assert seen == [0.2, 0.5, 1.0]
    assert monotonic(None) is None


def test_transcribe_reports_segment_progress(monkeypatch: pytest.MonkeyPatch):
    install_fake_whisper(monkeypatch)
    seen: list[float] = []
    words = transcribe.transcribe(TEN_SECONDS, device="cpu", on_progress=seen.append)
    assert seen == [0.25, 0.5, 0.75, 1.0]
    assert [w.text for w in words] == ["w1", "w2", "w3", "w4"]


def test_cpu_retry_after_cuda_failure_does_not_go_backwards(monkeypatch: pytest.MonkeyPatch):
    install_fake_whisper(monkeypatch, cuda_fails_after=2)
    seen: list[float] = []
    words = transcribe.transcribe(TEN_SECONDS, on_progress=seen.append, log=lambda _m: None)
    assert seen == [0.25, 0.5, 0.75, 1.0]
    assert len(words) == 4  # only the CPU run's words, not the half CUDA run's
