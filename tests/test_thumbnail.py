import shutil
from pathlib import Path

import numpy as np
import pytest
from conftest import make_video

from clipwright.probe import probe
from clipwright.thumbnail import make_thumbnail, pick_time


def test_stillest_frame_near_the_loudest_moment():
    loud = np.full(100, -30.0)  # 10s at 0.1s hops
    loud[60] = -5.0  # loudest at 6.05s
    energy = np.full(100, 20.0, dtype=np.float32)  # 10 fps
    energy[61] = 30.0  # the blurriest frame is right at the peak
    energy[66] = 2.0   # a sharp one half a second later
    energy[20] = 0.0   # stiller, but far from the peak
    assert pick_time(loud, energy, 10.0) == 6.6


def test_no_audio_goes_a_third_in():
    energy = np.full(100, 5.0, dtype=np.float32)
    t = pick_time(None, energy, 9.0)
    assert abs(t - 3.0) <= 1.0


def test_stays_off_the_edges():
    loud = np.full(100, -30.0)
    loud[0] = 0.0  # loudest in the first 0.1s
    assert pick_time(loud, np.zeros(0, dtype=np.float32), 10.0) >= 0.5


def test_very_short_clip():
    assert 0 <= pick_time(np.array([-10.0]), np.zeros(3, dtype=np.float32), 0.6) <= 0.6


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_make_thumbnail_on_a_real_clip(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    make_video(src, duration=20)
    out = tmp_path / "clip.jpg"
    t = make_thumbnail(src, out)
    assert 0.5 <= t <= 19.5
    shot = probe(out)
    assert shot.width == 720 and out.stat().st_size > 1000
