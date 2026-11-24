import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipwright.speaker import Face, Faces, faces_in, mouth_box, mouth_change, read_faces


def test_mouth_box_is_lower_middle_of_face():
    assert mouth_box((100, 50, 80, 90)) == (120, 110, 40, 30)
    # tiny boxes still give at least one pixel
    assert mouth_box((0, 0, 1, 1))[2:] == (1, 1)


def test_mouth_change_only_looks_at_the_mouth():
    prev = np.zeros((90, 80), dtype=np.uint8)
    cur = prev.copy()
    cur[0:20, :] = 200  # forehead changes, mouth doesn't
    assert mouth_change(cur, prev, (0, 0, 80, 90)) == 0.0
    cur[60:90, 20:60] = 100  # the whole mouth area changes by 100
    assert mouth_change(cur, prev, (0, 0, 80, 90)) == pytest.approx(100.0)
    # first frame: nothing to compare with
    assert mouth_change(cur, None, (0, 0, 80, 90)) == 0.0


def test_head_movement_is_not_talking():
    # a textured face that shifts 3 px between frames: the mouth area changes a lot, but so does
    # the eye band, so it shouldn't count as talking
    rng = np.random.default_rng(1)
    face = rng.integers(0, 255, (90, 80), dtype=np.uint8)
    prev = np.zeros((100, 100), dtype=np.uint8)
    cur = prev.copy()
    prev[0:90, 0:80] = face
    cur[0:90, 3:83] = face
    assert mouth_change(cur, prev, (0, 0, 80, 90)) < 3
    # the jaw moving on its own still counts
    talk = prev.copy()
    talk[62:90, 20:60] = rng.integers(0, 255, (28, 40), dtype=np.uint8)
    assert mouth_change(talk, prev, (0, 0, 80, 90)) > 50


def test_missing_opencv_names_the_extra(monkeypatch: pytest.MonkeyPatch):
    import sys

    from clipwright.speaker import haar_detector

    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match=r"clipwright\[speaker\]"):
        haar_detector()


def test_faces_in_scales_boxes_to_fractions():
    frame = np.zeros((100, 200), dtype=np.uint8)
    faces = faces_in(frame, None, lambda g: [(50, 10, 40, 50)])
    assert faces == [Face(0.25, 0.1, 0.2, 0.5, 0.0)]
    assert faces[0].centre == pytest.approx(0.35)
    assert faces[0].area == pytest.approx(0.1)


def test_faces_duration():
    assert Faces(5.0, [[], [], []]).duration == pytest.approx(0.6)


def test_haar_detector_finds_nothing_on_a_flat_frame():
    pytest.importorskip("cv2")
    from clipwright.speaker import haar_detector

    assert list(haar_detector()(np.full((270, 480), 128, dtype=np.uint8))) == []


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_read_faces_measures_mouth_movement(tmp_path: Path):
    # a patch flickers where the left face's mouth is (a "talking" mouth), everything else is
    # still. A fake detector reports one face on each side; only the left one should show mouth
    # activity. The patch stays below the left face's eye band, which is taken off.
    src = tmp_path / "two.mp4"
    # flips between dark and light every 0.2 s, offset so 5 fps samples land mid-flip-period
    flicker = "geq=lum='if(mod(floor(T*5+0.5),2),230,30)':cb=128:cr=128"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=25:d=2",
                    "-f", "lavfi", "-i", "color=c=white:s=120x90:r=25:d=2",
                    "-filter_complex", f"[1]{flicker}[f];[0][f]overlay=100:200",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    boxes = [(40, 50, 160, 160), (280, 50, 160, 160)]  # in the 480x270 analysis frame
    faces = read_faces(src, 0, 640, 360, detect=lambda g: boxes)
    assert faces.fps == 5.0
    assert len(faces.frames) == 10
    assert all(len(f) == 2 for f in faces.frames)
    left = [f[0].mouth for f in faces.frames[1:]]
    right = [f[1].mouth for f in faces.frames[1:]]
    assert min(left) > 50
    assert max(right) < 2
    # a window in the middle of the video
    part = read_faces(src, 0, 640, 360, start=1.0, duration=0.6, detect=lambda g: boxes)
    assert len(part.frames) == 3


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_read_faces_reports_ffmpeg_errors(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="face analysis failed"):
        read_faces(bad, 0, 640, 360, detect=lambda g: [])
