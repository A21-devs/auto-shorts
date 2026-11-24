"""Builds a synthetic video with ffmpeg and runs the whole pipeline on it (no whisper)."""

import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright.pipeline import Job, run
from clipwright.probe import probe
from clipwright.render import RenderOptions
from clipwright.segment import Options

pytestmark = [pytest.mark.ffmpeg, pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")]


def test_pipeline_finds_the_loud_part_and_renders_vertical(tmp_path: Path):
    src = tmp_path / "long.mp4"
    make_video(src)
    job = Job(source=src, out_dir=tmp_path / "out", clips=Options(count=1, min_len=15, max_len=25),
              render=RenderOptions(encoder="libx264"), captions=False)
    events: list[tuple[str, float]] = []
    manifest = json.loads(run(job, log=lambda _m: None, progress=lambda s, f: events.append((s, f))).read_text())

    # no captions here, so no transcribe stage
    stages = list(dict.fromkeys(s for s, _ in events))
    assert stages == ["audio", "scenes", "render", "done"]
    for stage in stages:
        fractions = [f for s, f in events if s == stage]
        assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert events[-1] == ("done", 1.0)

    clip = manifest["clips"][0]
    assert clip["start"] <= 45 and clip["end"] >= 50
    assert manifest["render"]["encoder"] == "libx264"
    assert manifest["words"] is None

    out = probe(tmp_path / "out" / clip["file"])
    assert (out.width, out.height) == (1080, 1920)
    assert 15 <= out.duration <= 26
    assert out.has_audio


@pytest.mark.parametrize("captions", [False, True])
def test_relative_source_path_from_the_video_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, captions: bool):
    make_video(tmp_path / "long.mp4")
    srt = None
    if captions:
        # subtitles instead of whisper, so the .ass path through render() (cwd = output folder) is covered
        srt = tmp_path / "long.srt"
        cues = [f"{i + 1}\n00:00:{i * 5:02d},000 --> 00:00:{i * 5 + 4:02d},000\nline {i}\n" for i in range(12)]
        srt.write_text("\n".join(cues), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # render runs ffmpeg inside the output folder, so a relative input used to point at out/long.mp4
    job = Job(source=Path("long.mp4"), out_dir=Path("out"), clips=Options(count=1, min_len=15, max_len=25),
              render=RenderOptions(encoder="libx264"), captions=captions, srt=srt)
    manifest = json.loads(run(job, log=lambda _m: None).read_text())
    assert Path(manifest["source"]) == tmp_path / "long.mp4"
    clip = manifest["clips"][0]
    assert (tmp_path / "out" / clip["file"]).stat().st_size > 0
    assert bool(clip["subtitles"]) == captions
