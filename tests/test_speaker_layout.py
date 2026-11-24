import shutil
import subprocess
from pathlib import Path

import pytest

from clipwright import render, speaker
from clipwright.cli import main
from clipwright.probe import MediaInfo
from clipwright.render import speaker_crop_x, video_filter
from clipwright.server import BadRequest, parse_job
from clipwright.speaker import Face, Faces

WIDE = MediaInfo(Path("talk.mp4"), 60, 1920, 1080, 30, True, 0)


def _podcast(script: str) -> Faces:
    """Two people at 0.25 and 0.75, one letter per second: L or R talks."""
    frames = []
    for who in script:
        for _ in range(5):
            frames.append([Face(0.15, 0.2, 0.2, 0.3, 15.0 if who == "L" else 0.3),
                           Face(0.65, 0.2, 0.2, 0.3, 15.0 if who == "R" else 0.3)])
    return Faces(5.0, frames)


def test_speaker_filter_uses_the_expression():
    g = video_filter("speaker", None, 30, 60, crop_x="clip(0.5*iw-ow/2,0,iw-ow)")
    assert "crop=w=1080:h=1920:x='clip(0.5*iw-ow/2,0,iw-ow)'" in g
    assert video_filter("speaker", None, 30, 60) == video_filter("crop", None, 30, 60)


def test_portrait_source_skips_face_detection():
    info = MediaInfo(Path("nope.mp4"), 60, 1080, 1920, 30, True, 0)
    assert speaker_crop_x(info, 0, 30) is None  # would fail if it tried to read the file


def test_speaker_crop_cuts_between_people(monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[float, float]] = []

    def fake(path: Path, stream: int, w: int, h: int, start: float, duration: float) -> Faces:
        seen.append((start, duration))
        return _podcast("LLLLRRRR")

    monkeypatch.setattr(speaker, "read_faces", fake)
    expr = speaker_crop_x(WIDE, 10.0, 18.0)
    assert seen == [(10.0, 8.0)]
    assert expr is not None and "gte(t,4.000)" in expr  # a hard cut, not a ramp
    # jump cuts removed 1.0-2.0: the cut moves to 3.0 and stays a cut
    expr = speaker_crop_x(WIDE, 10.0, 18.0, keep=[(0.0, 1.0), (2.0, 8.0)])
    assert expr is not None and "gte(t,3.000)" in expr and "/0." not in expr


def test_cli_needs_opencv_for_speaker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(speaker, "available", lambda: False)
    assert main([str(video), "--layout", "speaker"]) == 2
    assert "clipwright[speaker]" in capsys.readouterr().err


def test_server_takes_speaker_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(speaker, "available", lambda: True)
    assert parse_job({"video": str(video), "layout": "speaker"}).render.layout == "speaker"
    monkeypatch.setattr(speaker, "available", lambda: False)
    with pytest.raises(BadRequest, match="OpenCV"):
        parse_job({"video": str(video), "layout": "speaker"})


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_render_speaker_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "wide.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=6",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    monkeypatch.setattr(speaker, "read_faces", lambda *a: _podcast("LLLRRR"))
    info = MediaInfo(src, 6, 1280, 720, 25, False, 0)
    out = tmp_path / "out.mp4"
    render.render(info, 0, 6, out, render.RenderOptions(layout="speaker", encoder="libx264"))
    size = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                           str(out)], capture_output=True, text=True, check=True).stdout.strip()
    assert size == "1080,1920"
