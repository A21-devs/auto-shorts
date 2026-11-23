"""Picks the best clip windows from per-second signals.

Everything here is pure (numbers in, numbers out) so it's easy to test.

Signals, all on a timeline of `hop` seconds:
- loudness: z-scored dB, how loud this moment is compared to the rest of the video
- speech: words per second from the transcript (0 if no transcript)
- cuts: scene cut timestamps

A window's score rewards sustained energy, a strong peak (something happens), talking,
and a good first few seconds (the hook). Clips are then snapped to quiet moments or
sentence ends so they don't start or stop mid-word, and picked greedily without overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Clip:
    start: float
    end: float
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Signals:
    hop: float
    loudness: FloatArray            # z-scored
    speech: FloatArray | None = None  # words/sec per hop
    cuts: tuple[float, ...] = ()
    boundaries: tuple[float, ...] = ()  # good cut points: silence midpoints, sentence ends

    @property
    def duration(self) -> float:
        return len(self.loudness) * self.hop


@dataclass(frozen=True)
class Options:
    count: int = 6
    min_len: float = 20.0
    max_len: float = 55.0
    min_gap: float = 5.0          # seconds between picked clips
    hook_len: float = 3.0
    snap: float = 2.5             # how far a boundary may move a start/end


def _window_score(sig: Signals, a: int, b: int, hook: int) -> tuple[float, list[str]]:
    seg = sig.loudness[a:b]
    if len(seg) == 0:
        return float("-inf"), []
    mean = float(np.mean(seg))
    peak = float(np.percentile(seg, 95))
    head = float(np.mean(sig.loudness[a:a + hook])) if hook > 0 else 0.0
    score = 1.0 * mean + 0.6 * peak + 0.4 * head
    reasons: list[str] = []
    if peak > 1.5:
        reasons.append("loud moment")
    if head > 0.8:
        reasons.append("strong start")
    if sig.speech is not None:
        talk = float(np.mean(sig.speech[a:b]))
        score += 0.5 * min(talk, 3.0)
        if talk > 1.5:
            reasons.append("lots of talking")
    t0, t1 = a * sig.hop, b * sig.hop
    n_cuts = sum(1 for c in sig.cuts if t0 <= c < t1)
    if n_cuts:
        # some cuts = action, a ton of cuts = menu flicker / loading screens
        per_min = n_cuts / ((t1 - t0) / 60)
        score += 0.3 * min(per_min, 6) / 6 - (0.5 if per_min > 20 else 0)
        if 2 <= per_min <= 20:
            reasons.append("scene changes")
    return score, reasons


def _snap(t: float, points: tuple[float, ...], limit: float) -> float:
    best = t
    dist = limit
    for p in points:
        d = abs(p - t)
        if d <= dist:
            best, dist = p, d
    return best


def candidates(sig: Signals, opt: Options, step: float = 1.0) -> list[Clip]:
    total = sig.duration
    if total < opt.min_len:
        return [Clip(0.0, total, 0.0, ("whole video is shorter than one clip",))] if total > 0 else []
    lengths = sorted({opt.min_len, (opt.min_len + opt.max_len) / 2, opt.max_len})
    hook = int(round(opt.hook_len / sig.hop))
    out: list[Clip] = []
    t = 0.0
    while t + opt.min_len <= total:
        for length in lengths:
            if t + length > total:
                break
            start = _snap(t, sig.boundaries, opt.snap)
            end = _snap(t + length, sig.boundaries, opt.snap)
            end = min(end, total)
            if not (opt.min_len <= end - start <= opt.max_len) or start < 0:
                start, end = t, t + length
            a, b = int(start / sig.hop), int(end / sig.hop)
            score, reasons = _window_score(sig, a, b, hook)
            # tiny preference for longer clips when scores tie
            out.append(Clip(round(start, 2), round(end, 2), score + 0.01 * (end - start) / opt.max_len, tuple(reasons)))
        t += step
    return out


def pick(cands: list[Clip], opt: Options) -> list[Clip]:
    """Greedy best-first selection with no overlaps (plus min_gap), returned in time order."""
    chosen: list[Clip] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if len(chosen) >= opt.count:
            break
        if all(c.end + opt.min_gap <= o.start or c.start >= o.end + opt.min_gap for o in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.start)


def find_clips(sig: Signals, opt: Options) -> list[Clip]:
    return pick(candidates(sig, opt), opt)
