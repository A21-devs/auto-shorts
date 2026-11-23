"""Jump cuts: find the dead air inside a clip and work out what to keep.

Everything here works on clip-relative seconds (0 = clip start) and is pure, so the render step
only has to turn `Ranges` into trim/concat filters and the captions only have to be retimed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from . import audio
from .transcribe import Word

Ranges = list[tuple[float, float]]

HOP = 0.05           # 50 ms loudness resolution, fine enough to cut between words
MIN_SILENCE = 0.5    # pauses shorter than this are part of normal speech rhythm
PAD = 0.1            # audio kept on each side of a cut so words aren't clipped
MIN_KEEP = 0.2       # a kept piece shorter than this is a click or a breath, drop it
RANGE_DB = 25.0      # "silent" = this far below the clip's loud parts


def dead_air_threshold(db: npt.NDArray[np.float64]) -> float:
    """Level below which a stretch counts as dead air, relative to this clip's own speech level.

    An absolute -40 dB misses pauses over room noise or quiet music, and the 10th percentile
    (audio.quiet_threshold) always finds "silence" even in a clip with constant game sound.
    25 dB under the loud parts (90th percentile) catches real pauses and leaves music alone.

    The percentile only looks at windows above the digital-silence clamp, otherwise a clip that's
    mostly waiting with a few seconds of talk gets the silence level as its "loud" level. The
    result stays above the clamp, since silences() needs windows strictly below it.
    """
    floor = audio.SILENCE_DB + 1.0
    sound = db[db > floor]
    if len(sound) == 0:
        return floor
    return max(floor, float(np.percentile(sound, 90)) - RANGE_DB)


def dead_air(samples: npt.NDArray[np.float32], sample_rate: int = audio.SAMPLE_RATE,
             min_silence: float = MIN_SILENCE) -> Ranges:
    """Pauses of min_silence+ seconds in a clip's audio, as clip-relative (start, end)."""
    db = audio.loudness_curve(samples, sample_rate, hop=HOP)
    return audio.silences(db, HOP, dead_air_threshold(db), min_len=min_silence)


def keep_ranges(length: float, pauses: Sequence[tuple[float, float]], pad: float = PAD,
                min_keep: float = MIN_KEEP) -> Ranges:
    """Parts of [0, length] to keep after cutting out each pause (minus `pad` on both sides).

    A pause touching the start or end of the clip is trimmed without padding on that side,
    there's no word there to protect.
    """
    keep: Ranges = []
    pos = 0.0
    for a, b in sorted(pauses):
        a, b = max(0.0, a), min(length, b)
        cut_a = a if a <= 0 else a + pad
        cut_b = b if b >= length else b - pad
        if cut_b - cut_a <= 0 or cut_b <= pos:
            continue
        if cut_a > pos:
            keep.append((pos, cut_a))
        pos = max(pos, cut_b)
    if pos < length:
        keep.append((pos, length))
    kept = [(round(a, 3), round(b, 3)) for a, b in keep if b - a >= min_keep]
    return kept


def kept_duration(ranges: Ranges) -> float:
    return sum(b - a for a, b in ranges)


def retime(t: float, ranges: Ranges) -> float:
    """Clip time -> output time. A time inside a removed part maps to the cut point."""
    out = 0.0
    for a, b in ranges:
        if t < a:
            return out
        if t <= b:
            return out + (t - a)
        out += b - a
    return out


def retime_words(words: Sequence[Word], clip_start: float, ranges: Ranges) -> list[Word]:
    """Words (absolute times) -> words on the jump-cut output timeline (0 = first kept frame).

    Words that fall entirely inside a removed part are dropped: their audio is gone, and a
    zero-length word would still show up (unhighlighted) in its caption line.
    """
    out = []
    for w in words:
        a = retime(w.start - clip_start, ranges)
        b = retime(w.end - clip_start, ranges)
        if b - a > 1e-6 or w.end <= w.start:
            out.append(replace(w, start=round(a, 3), end=round(b, 3)))
    return out
