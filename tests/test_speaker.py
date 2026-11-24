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


# --- who to show ---------------------------------------------------------------------------------

from clipwright.speaker import clear_speaker, speaker_path, talking, tracks  # noqa: E402


def _face(centre: float, mouth: float, size: float = 0.2) -> Face:
    return Face(centre - size / 2, 0.2, size, size * 1.6, mouth)


def _podcast(script: str, fps: float = 5.0) -> Faces:
    """Two people at 0.25 and 0.75. `script` has one letter per second: L or R talks, - nobody."""
    frames = []
    for who in script:
        for _ in range(int(fps)):
            frames.append([_face(0.25, 15.0 if who == "L" else 0.3), _face(0.75, 15.0 if who == "R" else 0.3)])
    return Faces(fps, frames)


def test_tracks_group_faces_and_drop_rare_hits():
    f = _podcast("LLLLLLLLLL")
    f.frames[3].append(_face(0.5, 40.0))  # one false hit in the middle
    people = tracks(f)
    assert [round(t.centre, 2) for t in people] == [0.25, 0.75]
    assert all(t.presence == 1.0 for t in people)
    assert tracks(Faces(5.0, [[], []])) == []


def test_short_wide_shot_keeps_its_people():
    # edited podcast: 4 s wide shot (left talking), then a 14 s close-up of the left person
    wide = [[_face(0.25, 15.0, 0.08), _face(0.75, 0.3, 0.08)]] * 20
    close = [[_face(0.5, 15.0, 0.3)]] * 70
    f = Faces(5.0, wide + close)
    assert [round(t.centre, 2) for t in tracks(f)] == [0.25, 0.5, 0.75]
    keys = speaker_path(f, window=0.32)
    assert keys == [(0.0, 0.25), (4.0, 0.25), (4.0, 0.5), (18.0, 0.5)]
    # a face seen for under 2 s and under 30% of the time is still dropped
    f.frames[30] = f.frames[30] + [_face(0.9, 30.0)]
    assert len(tracks(f)) == 3


def test_talking_scores_per_second():
    scores = talking(_podcast("LR"), tracks(_podcast("LR")))
    assert scores[0][0] > 10 > scores[0][1]
    assert scores[1][1] > 10 > scores[1][0]


def test_clear_speaker_needs_a_lead():
    assert clear_speaker([15.0, 0.3]) == 0
    assert clear_speaker([0.3, 15.0]) == 1
    assert clear_speaker([10.0, 9.0]) is None  # both talking: crosstalk, keep the shot
    assert clear_speaker([1.0, 0.0]) is None   # nobody really talking
    assert clear_speaker([]) is None


def test_path_cuts_between_speakers():
    keys = speaker_path(_podcast("LLLLLLRRRRRRLLLLLL"), window=0.32)
    # holds on the left face, hard cut at 6 s, back at 12 s, both centres clamped into reach
    assert keys == [(0.0, 0.25), (6.0, 0.25), (6.0, 0.75), (12.0, 0.75), (12.0, 0.25), (18.0, 0.25)]


def test_path_ignores_short_interjections():
    # R says "yeah" for one second: no cut away and back. R then takes over at 9 s
    keys = speaker_path(_podcast("LLLLRLLLLRRRRR"), window=0.32)
    assert keys == [(0.0, 0.25), (9.0, 0.25), (9.0, 0.75), (14.0, 0.75)]


def test_path_waits_min_shot_between_cuts():
    # R takes over at 4 s, L answers 2 s later, L only gets the shot once 3 s have passed
    keys = speaker_path(_podcast("LLLLRRLLLLL"), window=0.32, min_shot=3.0)
    assert keys == [(0.0, 0.25), (4.0, 0.25), (4.0, 0.75), (7.0, 0.75), (7.0, 0.25), (11.0, 0.25)]


def test_path_ignores_a_takeover_cut_short_by_the_end():
    assert speaker_path(_podcast("LLLLLR"), window=0.32) == [(0.0, 0.25), (6.0, 0.25)]


def test_path_starts_on_the_biggest_face_when_nobody_talks():
    f = Faces(5.0, [[_face(0.25, 0.0, size=0.1), _face(0.7, 0.0, size=0.25)]] * 20)
    keys = speaker_path(f, window=0.32)
    assert keys == [(0.0, 0.7), (4.0, 0.7)]


def test_path_clamps_to_the_frame_and_handles_no_faces():
    f = Faces(5.0, [[_face(0.05, 15.0)]] * 10)
    assert speaker_path(f, window=0.32)[0] == (0.0, pytest.approx(0.16))
    assert speaker_path(Faces(5.0, [[]] * 10), window=0.32) == [(0.0, 0.5)]
    assert speaker_path(f, window=1.0) == [(0.0, 0.5)]
