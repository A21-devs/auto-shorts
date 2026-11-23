"""Builds a synthetic video with ffmpeg and runs the whole pipeline on it (no whisper)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autoshorts.pipeline import Job, run
from autoshorts.probe import probe
from autoshorts.render import RenderOptions
from autoshorts.segment import Options

pytestmark = [pytest.mark.ffmpeg, pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")]


def make_video(path: Path) -> None:
    # 90s: quiet tone, then a loud burst from 40s to 55s, then quiet again
    audio = "aevalsrc='0.02*sin(2*PI*220*t) + if(between(t,40,55), 0.8*sin(2*PI*660*t), 0)':s=44100:d=90"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=90",
         "-f", "lavfi", "-i", audio, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )


def test_pipeline_finds_the_loud_part_and_renders_vertical(tmp_path: Path):
    src = tmp_path / "long.mp4"
    make_video(src)
    job = Job(source=src, out_dir=tmp_path / "out", clips=Options(count=1, min_len=15, max_len=25),
              render=RenderOptions(encoder="libx264"), captions=False)
    manifest = json.loads(run(job, log=lambda _m: None).read_text())

    clip = manifest["clips"][0]
    assert clip["start"] <= 45 and clip["end"] >= 50

    out = probe(tmp_path / "out" / clip["file"])
    assert (out.width, out.height) == (1080, 1920)
    assert 15 <= out.duration <= 26
    assert out.has_audio
