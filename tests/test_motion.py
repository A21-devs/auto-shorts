import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from autoshorts.motion import Motion, focus, read_motion, small_size


def test_small_size_keeps_aspect_and_even_height():
    assert small_size(1920, 1080) == (96, 54)
    assert small_size(1080, 1920) == (96, 170)
    assert small_size(640, 480, target=64) == (64, 48)
    with pytest.raises(ValueError):
        small_size(0, 1080)


def _motion(rows: list[list[float]], fps: float = 5.0) -> Motion:
    return Motion(fps, np.array(rows, dtype=np.float32))


def test_energy_is_mean_over_columns():
    m = _motion([[0, 0, 0, 0], [4, 0, 0, 0], [2, 2, 2, 2]])
    assert m.energy.tolist() == [0, 1, 2]
    assert m.duration == pytest.approx(0.6)
    assert _motion([], fps=5).energy.size == 0


def test_focus_points_at_the_moving_columns():
    # 10 columns, motion only in columns 7-8 for 1 second at 5 fps
    row = [0.0] * 10
    row[7] = row[8] = 30.0
    f = focus(_motion([row] * 5), window=1.0)
    assert len(f) == 1
    assert f[0] == pytest.approx(0.8, abs=1e-6)


def test_uniform_motion_has_no_opinion():
    # a pan moves every column the same amount: nothing stands out, so stay put (NaN)
    f = focus(_motion([[20.0] * 10] * 5))
    assert math.isnan(f[0])


def test_still_windows_have_no_opinion():
    row = [0.0] * 10
    row[1] = 2.0  # mean 0.2, under the still threshold
    f = focus(_motion([row] * 5))
    assert math.isnan(f[0])


def test_focus_counts_a_partial_last_window():
    row = [0.0] * 4
    row[0] = 40.0
    f = focus(_motion([row] * 7), window=1.0)  # 5 + 2 frames
    assert len(f) == 2
    assert f[1] == pytest.approx(0.125)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_read_motion_finds_a_box_moving_on_the_right(tmp_path: Path):
    src = tmp_path / "box.mp4"
    graph = ("color=c=black:s=320x180:r=10:d=4[bg];color=c=white:s=40x40:r=10:d=4[box];"
             "[bg][box]overlay=x='240+20*sin(2*PI*t)':y=70:eval=frame,format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)

    m = read_motion(src, 0, 320, 180, fps=5)
    assert m.columns.shape == (20, 96)
    assert m.columns[0].sum() == 0
    f = focus(m)
    assert len(f) == 4
    # the box lives in x = 220..300 of 320, centre around 0.8
    assert np.nanmin(f) > 0.65 and np.nanmax(f) < 0.95

    part = read_motion(src, 0, 320, 180, start=1.0, duration=2.0, fps=5)
    assert part.columns.shape[0] == 10
    assert read_motion(src, 0, 320, 180, start=10.0, fps=5).columns.shape == (0, 96)


def test_placeholder_row_does_not_hide_the_first_window():
    # real motion mean is 1.2 (just over STILL). Averaging in the zero row 0 would make it 0.96
    row = [0.0] * 10
    row[2] = 12.0
    f = focus(_motion([[0.0] * 10] + [row] * 9))
    assert f[0] == pytest.approx(0.25)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_ffmpeg_is_killed_when_reading_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=60",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    import autoshorts.motion as motion

    started: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        p = real_popen(*args, **kwargs)
        started.append(p)
        return p

    calls = 0
    real_frombuffer = np.frombuffer

    def boom(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return real_frombuffer(*args, **kwargs)

    monkeypatch.setattr(motion.subprocess, "Popen", spy)
    monkeypatch.setattr(motion.np, "frombuffer", boom)
    with pytest.raises(KeyboardInterrupt):
        read_motion(src, 0, 320, 180)
    assert calls == 3
    assert started and started[0].poll() is not None
