"""Renders a synthetic talk-pause-talk video with --jumpcut and checks the output for real."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from autoshorts.pipeline import Job, plan_cuts, run
from autoshorts.probe import probe
from autoshorts.render import RenderOptions
from autoshorts.segment import Options

pytestmark = [pytest.mark.ffmpeg, pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")]


def _stream_duration(path: Path, kind: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries",
                          "stream=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def test_jumpcut_removes_the_pauses_and_keeps_av_in_sync(tmp_path: Path):
    src = tmp_path / "talk.mp4"
    # tone, then 3s of silence at 8-11s and 14-17s, 24s total
    tone = "0.3*sin(2*PI*300*t)*not(between(t,8,11))*not(between(t,14,17))"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=24",
                    "-f", "lavfi", "-i", f"aevalsrc='{tone}':s=44100:d=24",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    info = probe(src)
    keep = plan_cuts(info, 0, 24)
    assert keep is not None and len(keep) == 3
    assert keep[0][1] == pytest.approx(8.1, abs=0.1) and keep[1][0] == pytest.approx(10.9, abs=0.1)

    job = Job(source=src, out_dir=tmp_path / "out", clips=Options(count=1, min_len=20, max_len=24),
              render=RenderOptions(encoder="libx264", jumpcut=True), captions=False)
    manifest = json.loads(run(job, log=lambda _m: None).read_text())
    clip = manifest["clips"][0]
    out = tmp_path / "out" / clip["file"]
    v, a = _stream_duration(out, "v:0"), _stream_duration(out, "a:0")
    expected = clip["end"] - clip["start"] - 2 * (3.0 - 0.2)
    assert clip["length"] == pytest.approx(expected, abs=0.2)
    assert v == pytest.approx(clip["length"], abs=0.1)
    assert abs(v - a) < 0.1
    assert manifest["render"]["jumpcut"] is True


def test_no_audio_means_no_cuts(tmp_path: Path):
    src = tmp_path / "mute.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=3",
                    "-c:v", "libx264", "-preset", "ultrafast", str(src)], check=True)
    assert plan_cuts(probe(src), 0, 3) is None
