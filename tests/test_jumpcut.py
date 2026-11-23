import numpy as np
import pytest

from autoshorts.captions import build_ass
from autoshorts.jumpcut import (
    dead_air,
    dead_air_threshold,
    keep_ranges,
    kept_duration,
    retime,
    retime_words,
)
from autoshorts.transcribe import Word

SR = 16000


def _tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _quiet(seconds: float, amp: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def test_dead_air_finds_the_pause_between_speech():
    pcm = np.concatenate([_tone(2), _quiet(1.0), _tone(2), _quiet(0.3), _tone(1)])
    pauses = dead_air(pcm, SR)
    assert len(pauses) == 1  # the 0.3s gap is normal rhythm
    a, b = pauses[0]
    assert a == pytest.approx(2.0, abs=0.06) and b == pytest.approx(3.0, abs=0.06)


def test_room_noise_still_counts_as_a_pause():
    # -50 dBFS hiss in the pause, above the old absolute floor tricks but far under the speech
    pcm = np.concatenate([_tone(2), _quiet(1.0, amp=0.003), _tone(2)])
    assert len(dead_air(pcm, SR)) == 1


def test_constant_music_has_no_dead_air():
    pcm = np.concatenate([_tone(3, 0.3), _tone(1, 0.15), _tone(3, 0.3)])  # a 6 dB dip is not a pause
    assert dead_air(pcm, SR) == []


def test_threshold_stays_above_the_silence_clamp():
    assert dead_air_threshold(np.full(10, -50.0)) == -49.0
    assert dead_air_threshold(np.zeros(0)) == -49.0


def test_quiet_mic_still_gets_cuts():
    # speech peaking around -33 dBFS: 25 dB under that is below the clamp
    pcm = np.concatenate([_tone(2, 0.03), _quiet(1.0), _tone(2, 0.03)])
    assert len(dead_air(pcm, SR)) == 1


def test_mostly_silent_clip_uses_the_speech_level():
    pcm = np.concatenate([_tone(2), _quiet(20), _tone(2), _quiet(20)])
    pauses = dead_air(pcm, SR)
    assert [(round(a), round(b)) for a, b in pauses] == [(2, 22), (24, 44)]


def test_keep_ranges_pads_both_sides():
    assert keep_ranges(10, [(4.0, 5.0)], pad=0.1) == [(0.0, 4.1), (4.9, 10.0)]


def test_pause_at_the_edges_is_trimmed_without_padding():
    assert keep_ranges(10, [(0.0, 1.0), (9.0, 10.0)], pad=0.1) == [(0.9, 9.1)]


def test_short_leftover_pieces_are_dropped():
    # two pauses 0.1s apart leave a 0.1s sliver (a click) between them
    assert keep_ranges(10, [(3.0, 4.0), (4.1, 5.0)], pad=0.0) == [(0.0, 3.0), (5.0, 10.0)]
    # with padding the piece between them grows to 0.45s and stays
    assert keep_ranges(10, [(3.0, 4.0), (4.25, 5.0)], pad=0.1) == [(0.0, 3.1), (3.9, 4.35), (4.9, 10.0)]


def test_pause_shorter_than_twice_the_pad_is_kept():
    assert keep_ranges(10, [(4.0, 4.15)], pad=0.1) == [(0.0, 10.0)]


def test_no_pauses_keeps_everything():
    assert keep_ranges(12.5, []) == [(0.0, 12.5)]
    assert kept_duration([(0.0, 4.1), (4.9, 10.0)]) == pytest.approx(9.2)


def test_retime_maps_across_cuts():
    ranges = [(0.0, 4.0), (5.0, 10.0)]
    assert retime(2.0, ranges) == 2.0
    assert retime(4.5, ranges) == 4.0     # inside the cut -> the cut point
    assert retime(6.0, ranges) == 5.0
    assert retime(-1.0, ranges) == 0.0
    assert retime(99.0, ranges) == 9.0


def test_retimed_words_build_captions_on_the_short_timeline():
    ranges = [(0.0, 4.1), (4.9, 10.0)]
    words = [Word(102.0, 102.5, "before"), Word(105.0, 105.4, "after"), Word(104.4, 104.6, "gone")]
    out = retime_words(words, 100.0, ranges)
    assert out[0] == Word(2.0, 2.5, "before")
    assert out[1] == Word(4.2, 4.6, "after")
    assert len(out) == 2  # "gone" was inside the cut
    ass = build_ass(out, 0.0, kept_duration(ranges))
    assert "BEFORE" in ass and "AFTER" in ass and "GONE" not in ass
