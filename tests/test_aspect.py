"""--aspect: square, feed and landscape output next to the 9:16 default."""

import argparse
import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright.captions import CaptionStyle, build_ass
from clipwright.cli import add_job_args, check_job_args
from clipwright.pipeline import Job, run
from clipwright.platforms import PLATFORMS, aspect_for, lengths
from clipwright.probe import probe
from clipwright.render import ASPECTS, RenderOptions, frame_size, video_filter
from clipwright.segment import Options
from clipwright.server import BadRequest, parse_job
from clipwright.transcribe import Word


def test_frame_sizes():
    assert frame_size("9:16") == (1080, 1920)
    assert frame_size("1:1") == (1080, 1080)
    assert frame_size("4:5") == (1080, 1350)
    assert frame_size("16:9") == (1920, 1080)
    with pytest.raises(ValueError, match="aspect must be one of"):
        frame_size("21:9")


@pytest.mark.parametrize("aspect", list(ASPECTS))
@pytest.mark.parametrize("layout", ["blur", "crop", "fit", "follow"])
def test_filter_uses_the_frame_size(aspect: str, layout):
    w, h = frame_size(aspect)
    g = video_filter(layout, None, 30, 60, "0" if layout == "follow" else None, (w, h))
    assert f"{w}:{h}" in g
    if aspect != "9:16":
        assert "1080:1920" not in g


def test_platform_picks_the_frame_unless_aspect_is_given():
    assert aspect_for(None, None) == "9:16"
    assert aspect_for("shorts", None) == "9:16"
    assert aspect_for("youtube", None) == "16:9"
    assert aspect_for("linkedin", None) == "1:1"
    assert aspect_for("youtube", "9:16") == "9:16"
    assert aspect_for("nope", None) == "9:16"


def test_new_platform_lengths():
    assert lengths("youtube", None, None) == (60, 300)
    assert lengths("x", None, None) == (15, 120)
    with pytest.raises(ValueError, match="at most 140"):
        lengths("x", None, 150)
    assert all(p.aspect in ASPECTS for p in PLATFORMS.values())


def _cli_args(*argv: str) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_job_args(p)
    args = p.parse_args(list(argv))
    assert check_job_args(args) is None
    return args


def test_cli_fills_in_the_aspect():
    assert _cli_args().aspect == "9:16"
    assert _cli_args("--platform", "youtube").aspect == "16:9"
    assert _cli_args("--platform", "youtube", "--aspect", "1:1").aspect == "1:1"


def test_server_takes_an_aspect(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert parse_job({"video": str(video)}).render.aspect == "9:16"
    assert parse_job({"video": str(video), "platform": "x"}).render.aspect == "16:9"
    assert parse_job({"video": str(video), "aspect": "4:5"}).render.aspect == "4:5"
    with pytest.raises(BadRequest, match="aspect"):
        parse_job({"video": str(video), "aspect": "3:2"})


def test_captions_scale_to_the_frame():
    base = CaptionStyle()
    assert base.for_frame(1080, 1920) == base
    wide = base.for_frame(1920, 1080)
    assert wide.size < base.size and wide.size >= round(base.size * 0.65)
    assert wide.margin_v < base.margin_v and wide.max_chars > base.max_chars
    square = base.for_frame(1080, 1080)
    assert square.margin_v < base.margin_v
    ass = build_ass([Word(0.0, 0.5, "hi")], 0, 1, wide, width=1920, height=1080)
    assert "PlayResX: 1920\nPlayResY: 1080" in ass


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("aspect", ["1:1", "16:9"])
def test_renders_at_the_chosen_aspect(tmp_path: Path, aspect: str):
    src = tmp_path / "long.mp4"
    make_video(src)
    srt = tmp_path / "long.srt"
    srt.write_text("\n".join(f"{i + 1}\n00:00:{i * 5:02d},000 --> 00:00:{i * 5 + 4:02d},000\nline {i}\n"
                             for i in range(12)), encoding="utf-8")
    job = Job(source=src, out_dir=tmp_path / "out", clips=Options(count=1, min_len=15, max_len=20),
              render=RenderOptions(encoder="libx264", aspect=aspect), srt=srt)
    manifest = json.loads(run(job, log=lambda _m: None).read_text())
    clip = manifest["clips"][0]
    assert manifest["render"]["aspect"] == aspect
    out = probe(tmp_path / "out" / clip["file"])
    assert (out.width, out.height) == frame_size(aspect)
    assert "PlayResX: %d" % frame_size(aspect)[0] in (tmp_path / "out" / clip["subtitles"]).read_text()
