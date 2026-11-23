import numpy as np

from autoshorts.audio import SILENCE_DB, loudness_curve, quiet_threshold, silences, zscore


def test_loudness_of_full_scale_sine_is_about_minus_3db():
    sr = 16000
    t = np.arange(sr * 2) / sr
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    db = loudness_curve(sine, sr, hop=0.5)
    assert len(db) == 4
    assert np.allclose(db, -3.01, atol=0.1)


def test_digital_silence_is_clamped():
    db = loudness_curve(np.zeros(16000, dtype=np.float32), 16000, hop=0.5)
    assert np.all(db == SILENCE_DB)


def test_too_short_input():
    assert len(loudness_curve(np.zeros(10, dtype=np.float32), 16000, hop=0.5)) == 0


def test_silences_found_and_min_length_respected():
    db = np.array([-20, -60, -60, -20, -60, -20, -60, -60, -60], dtype=float)
    assert silences(db, hop=0.5, threshold_db=-40, min_len=1.0) == [(0.5, 1.5), (3.0, 4.5)]


def test_loud_video_still_gets_quiet_points():
    # constant game music around -18 dB with a few dips to -27, never below -40
    db = np.full(200, -18.0)
    db[50:53] = -27.0
    db[120:124] = -27.5
    assert silences(db, hop=0.5, min_len=1.0) == [(25.0, 26.5), (60.0, 62.0)]


def test_real_silence_keeps_minus_40_floor():
    db = np.concatenate([np.full(50, -20.0), np.full(10, -60.0), np.full(50, -20.0)])
    assert quiet_threshold(db) == -40.0


def test_zscore_constant_is_zero():
    assert np.all(zscore(np.full(5, 7.0)) == 0)
