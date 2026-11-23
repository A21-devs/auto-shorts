import numpy as np
import pytest

from autoshorts.segment import Clip, Options, Signals, candidates, find_clips, pick


def flat(seconds: float, hop: float = 0.5) -> np.ndarray:
    return np.zeros(int(seconds / hop))


def test_picks_the_loud_part():
    loud = flat(600)
    loud[400:440] = 3.0  # 200s-220s is way louder than the rest
    clips = find_clips(Signals(0.5, loud), Options(count=1, min_len=20, max_len=40))
    assert len(clips) == 1
    c = clips[0]
    assert c.start <= 200 and c.end >= 220


def test_clips_never_overlap_and_respect_gap():
    rng = np.random.default_rng(0)
    sig = Signals(0.5, rng.normal(size=2400))
    opt = Options(count=8, min_len=20, max_len=50, min_gap=5)
    clips = find_clips(sig, opt)
    assert len(clips) == 8
    for a, b in zip(clips, clips[1:], strict=False):
        assert a.end + opt.min_gap <= b.start


def test_lengths_stay_in_range():
    rng = np.random.default_rng(1)
    sig = Signals(0.5, rng.normal(size=1200), boundaries=tuple(float(x) for x in range(0, 600, 7)))
    opt = Options(count=5, min_len=15, max_len=45)
    for c in find_clips(sig, opt):
        assert opt.min_len <= c.duration <= opt.max_len + 1e-6


def test_snaps_to_boundaries():
    sig = Signals(0.5, flat(300), boundaries=(101.0, 131.5))
    cands = candidates(sig, Options(min_len=25, max_len=35, snap=2.5))
    assert any(c.start == 101.0 and c.end == 131.5 for c in cands)


def test_short_video_becomes_one_clip():
    clips = find_clips(Signals(0.5, flat(12)), Options(min_len=20))
    assert [(c.start, c.end) for c in clips] == [(0.0, 12.0)]


def test_empty_video():
    assert find_clips(Signals(0.5, flat(0)), Options()) == []


def test_pick_returns_time_order():
    cands = [Clip(100, 130, 5.0), Clip(10, 40, 9.0), Clip(200, 230, 7.0)]
    assert [c.start for c in pick(cands, Options(count=3))] == [10, 100, 200]


def test_talking_beats_silence():
    loud = flat(300)
    speech = flat(300)
    speech[100:160] = 3.0  # 50s-80s someone is talking
    clips = find_clips(Signals(0.5, loud, speech=speech), Options(count=1, min_len=20, max_len=30))
    assert clips[0].start == pytest.approx(50, abs=12)
    assert "lots of talking" in clips[0].reasons
