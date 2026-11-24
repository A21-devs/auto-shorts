import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from autoshorts.follow import camera_path, centre_expr, crop_x_expr, simplify
from autoshorts.probe import MediaInfo
from autoshorts.render import follow_crop_x, video_filter

NAN = float("nan")


def _f(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _centre_at(keys: list[tuple[float, float]], t: float) -> float:
    """Reference evaluation of centre_expr in Python (same formula ffmpeg sees)."""
    expr = centre_expr(keys).replace("clip(", "_clip(")
    return float(eval(expr, {"t": t, "_clip": lambda x, lo, hi: min(hi, max(lo, x)),
                             "gte": lambda a, b: float(a >= b)}))


def test_source_already_narrow_means_centre():
    assert camera_path(_f(0.1, 0.9), 1.0, window=1.0) == [(0.0, 0.5)]
    assert camera_path(_f(), 1.0, window=0.3) == [(0.0, 0.5)]


def test_starts_on_the_first_known_focus_and_stays_reachable():
    keys = camera_path(_f(NAN, 0.05, 0.05), 1.0, window=0.3)
    # 0.05 is too far left for a 0.3 wide crop, the leftmost reachable centre is 0.15
    assert keys[0] == (0.0, pytest.approx(0.15))
    assert all(c == pytest.approx(0.15) for _, c in keys)


def test_small_moves_inside_the_dead_zone_are_ignored():
    keys = camera_path(_f(0.5, 0.55, 0.45, 0.52), 1.0, window=0.3)
    assert len(keys) == 2 and keys[0][1] == keys[1][1]


def test_pan_speed_is_limited():
    keys = camera_path(_f(0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8), 1.0, window=0.3, max_speed=0.25)
    cs = [c for _, c in keys]
    assert cs[0] == pytest.approx(0.2)
    assert max(cs) <= 0.8 + 1e-6
    # never more than 0.25 per second between keyframes
    for (t0, c0), (t1, c1) in zip(keys, keys[1:], strict=False):
        assert abs(c1 - c0) <= 0.25 * (t1 - t0) + 1e-6
    # and it gets there in the end
    assert cs[-1] > 0.8 - 0.3 / 4


def test_nan_holds_position():
    keys = camera_path(_f(0.2, NAN, NAN, NAN), 1.0, window=0.3)
    assert keys == [(0.0, pytest.approx(0.2)), (3.5, pytest.approx(0.2))]


def test_one_odd_window_does_not_start_a_pan():
    keys = camera_path(_f(0.3, 0.3, 0.9, 0.3, 0.3), 1.0, window=0.3)
    assert all(c == pytest.approx(0.3) for _, c in keys)


def test_a_pan_recentres_instead_of_stopping_at_the_dead_zone_edge():
    keys = camera_path(_f(*([0.3] * 3 + [0.6] * 10)), 1.0, window=0.316)
    assert keys[0][1] == pytest.approx(0.3)
    # old behaviour parked at 0.6 - 0.316 / 4 = 0.521
    assert keys[-1][1] > 0.6 - 0.316 / 16 - 1e-6


def test_start_position_is_a_real_focus_value_not_a_blend():
    # windows 0 and 2 agree, window 1 is an outlier: start on 0.9, not on mean(0.9, 0.54) = 0.72
    keys = camera_path(_f(0.9, 0.54, 0.9, 0.9), 1.0, window=0.316)
    assert keys[0][1] == pytest.approx(1 - 0.316 / 2)


def test_a_still_second_mid_pan_does_not_end_the_pan():
    keys = camera_path(_f(*([0.3] * 3 + [0.6, NAN] + [0.6] * 8)), 1.0, window=0.316)
    assert keys[-1][1] > 0.6 - 0.316 / 16 - 1e-6
    # and it kept moving through the NaN second (t = 4.5)
    cs = dict(keys)
    assert 4.5 not in cs or cs[4.5] > cs.get(3.5, 0.3)


def test_two_known_values_are_not_averaged():
    keys = camera_path(_f(0.9, NAN, 0.54), 1.0, window=0.316)
    assert keys[0][1] == pytest.approx(1 - 0.316 / 2)


def test_simplify_drops_points_on_a_line():
    keys = [(0.0, 0.5), (1.0, 0.5), (2.0, 0.5), (3.0, 0.6), (4.0, 0.7), (5.0, 0.7)]
    assert simplify(keys) == [(0.0, 0.5), (2.0, 0.5), (4.0, 0.7), (5.0, 0.7)]


def test_centre_expr_matches_the_keyframes():
    keys = [(0.0, 0.3), (2.0, 0.3), (4.0, 0.7), (6.0, 0.5)]
    for t, want in [(0, 0.3), (1, 0.3), (3, 0.5), (4, 0.7), (5, 0.6), (6, 0.5), (60, 0.5)]:
        assert _centre_at(keys, t) == pytest.approx(want, abs=1e-4)
    assert centre_expr([(0.0, 0.42)]) == "0.4200"


def test_zero_length_segment_is_a_step_not_dropped():
    # a jump cut collapsed 2.0-3.0 to a single instant: the 0.1 move must still happen
    keys = [(0.0, 0.3), (2.0, 0.3), (2.0, 0.4), (4.0, 0.6)]
    assert _centre_at(keys, 1.0) == pytest.approx(0.3)
    assert _centre_at(keys, 2.5) == pytest.approx(0.45)
    assert _centre_at(keys, 5.0) == pytest.approx(0.6)


def test_follow_filter_uses_the_expression():
    g = video_filter("follow", "c.ass", 30, 60, crop_x=crop_x_expr([(0.0, 0.5), (1.0, 0.6)]))
    assert "crop=w=1080:h=1920:x='clip((" in g and g.endswith("ass=c.ass[vout]")
    # without an expression it's a plain centre crop
    assert video_filter("follow", None, 30, 60) == video_filter("crop", None, 30, 60)


def test_portrait_source_skips_the_motion_pass():
    info = MediaInfo(Path("nope.mp4"), 60, 1080, 1920, 30, True, 0)
    assert follow_crop_x(info, 0, 30) is None  # would fail if it tried to read the file


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_ffmpeg_accepts_the_crop_expression(tmp_path: Path):
    keys = camera_path(_f(0.2, 0.2, 0.9, 0.9, 0.9, 0.9), 1.0, window=1080 / 3413)
    g = video_filter("follow", None, 30, 60, crop_x=crop_x_expr(keys))
    proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=10:d=6",
                           "-filter_complex", g, "-map", "[vout]", "-f", "null", "-"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert not math.isnan(_centre_at(keys, 3.0))
