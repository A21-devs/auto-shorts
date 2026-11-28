import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import make_video

from clipwright.captions import build_ass
from clipwright.probe import MediaInfo, probe
from clipwright.render import RenderOptions, cover_command, render, render_cover
from clipwright.thumbnail import grab
from clipwright.transcribe import Word

INFO = MediaInfo(Path("in.mp4"), duration=600, width=1920, height=1080, fps=30, has_audio=True, video_stream=1)


def test_cover_has_no_captions_and_seeks_the_source():
    cmd = cover_command(INFO, 42.5, Path("c.jpg"), RenderOptions(layout="blur"), clip_time=2.5)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "ass=" not in graph
    assert graph.startswith("[0:1]setpts=PTS-STARTPTS+2.500/TB,")  # the probed stream, on the clip's clock
    assert graph.endswith("[vout]scale=720:-2[cover]")
    assert cmd[cmd.index("-ss") + 1] == "42.500"
    assert cmd[cmd.index("-frames:v") + 1] == "1"


def test_cover_keeps_the_follow_window_and_the_aspect():
    opts = RenderOptions(layout="follow", aspect="4:5")
    cmd = cover_command(INFO, 10.0, Path("c.jpg"), opts, crop_x="min(t*10,iw-ow)")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "x='min(t*10,iw-ow)'" in graph and "crop=w=1080:h=1350" in graph


def test_high_fps_sources_get_no_fps_filter():
    # the fps filter can swallow a lone frame, a cover never needs it
    info = MediaInfo(Path("in.mp4"), 600, 1920, 1080, 119.88, True, video_stream=0)
    graph = cover_command(info, 1.0, Path("c.jpg"), RenderOptions())
    assert "fps=" not in graph[graph.index("-filter_complex") + 1]


def _gray(jpg: Path) -> np.ndarray:
    info = probe(jpg)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(jpg), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(info.height, info.width).astype(np.float32)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_cover_matches_the_clip_without_the_caption(tmp_path: Path):
    src = tmp_path / "src.mp4"
    make_video(src, duration=20)
    info = probe(src)
    # a panning window, so a wrong clock would show up as a shifted frame
    opts = RenderOptions(layout="follow", encoder="libx264", normalize_audio=False)
    text = ["NUMBER", "SIXTEEN", "OF", "THE", "LIST"] * 2
    words = [Word(10.0 + i * 0.5, 10.4 + i * 0.5, w) for i, w in enumerate(text)]
    (tmp_path / "c.ass").write_text(build_ass(words, 10.0, 16.0), encoding="utf-8")
    crop_x = "min(t*40,iw-ow)"

    def fake_follow(*_args: object, **_kw: object) -> str:
        return crop_x

    clip = tmp_path / "c.mp4"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("clipwright.render.follow_crop_x", fake_follow)
        used = render(info, 10.0, 16.0, clip, opts, "c.ass", workdir=tmp_path)
    assert used == crop_x

    grab(clip, 3.0, tmp_path / "from_clip.jpg")
    render_cover(info, 13.0, tmp_path / "cover.jpg", opts, crop_x, clip_time=3.0)
    a, b = _gray(tmp_path / "from_clip.jpg"), _gray(tmp_path / "cover.jpg")
    assert a.shape == b.shape == (1280, 720)
    rows = np.abs(a - b).mean(axis=1)
    caption = rows > 12  # rows where the burned-in caption is
    assert caption.any(), "the clip frame should have a caption on it"
    # everywhere else the cover is the clip's frame, give or take two encodes
    assert float(rows[~caption].mean()) < 4.0
