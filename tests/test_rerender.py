import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright import pipeline
from clipwright.pipeline import Job, rerender_clip, run
from clipwright.probe import probe
from clipwright.render import RenderOptions
from clipwright.segment import Options
from clipwright.transcribe import Word, load_words, save_words


def fake_run(out: Path, clips: int = 2) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    data = {"source": str(out / "src.mp4"), "duration": 60.0, "render": {"layout": "fit"}, "words": None,
            "clips": [{"start": 10.0 * i, "end": 10.0 * i + 5, "score": 1.0, "reasons": [],
                       "file": f"src_clip{i:02d}.mp4", "subtitles": None} for i in range(1, clips + 1)]}
    (out / "src.mp4").write_bytes(b"source")
    for c in data["clips"]:
        (out / c["file"]).write_bytes(b"old clip")
    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest


def test_words_round_trip(tmp_path: Path):
    words = [Word(1.0, 1.4, "héllo"), Word(1.5, 2.0, "world.")]
    save_words(words, tmp_path / "w.json")
    assert load_words(tmp_path / "w.json") == words


def test_rerender_respects_the_platform_limit(tmp_path: Path):
    manifest = fake_run(tmp_path)
    data = json.loads(manifest.read_text())
    data["platform"], data["duration"] = "shorts", 120.0
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="at most 60s"):
        rerender_clip(manifest, 1, 0, 61, log=lambda _m: None)


@pytest.mark.parametrize("n", [0, 3, -1])
def test_bad_clip_number(tmp_path: Path, n: int):
    with pytest.raises(ValueError, match="doesn't exist"):
        rerender_clip(fake_run(tmp_path), n, 1, 10, log=lambda _m: None)


@pytest.mark.parametrize(("start", "end"), [(10, 10), (12, 11), (59.5, 80), (-5, 0.5)])
def test_too_short_after_clamping(tmp_path: Path, start: float, end: float):
    with pytest.raises(ValueError, match="at least"):
        rerender_clip(fake_run(tmp_path), 1, start, end, log=lambda _m: None)


def test_render_options_from_manifest():
    assert pipeline._render_options(None) == RenderOptions()
    assert pipeline._render_options({"layout": "crop", "from_the_future": 1}) == RenderOptions(layout="crop")


def test_failed_render_keeps_old_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = fake_run(tmp_path)

    def broken_render(_info, _a, _b, out: Path, *_args, **_kw):
        out.write_bytes(b"half a file")
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", broken_render)
    with pytest.raises(RuntimeError):
        rerender_clip(manifest, 1, 5, 20, log=lambda _m: None)
    assert (tmp_path / "src_clip01.mp4").read_bytes() == b"old clip"
    assert not list(tmp_path.glob("*.tmp*"))
    assert json.loads(manifest.read_text())["clips"][0]["start"] == 10.0


def fake_render(_info, _a, _b, out: Path, *_args, **_kw):
    out.write_bytes(b"new clip")


def test_old_manifest_drops_stale_captions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # manifests from before `words` existed: the clip had captions but there's no transcript to rebuild them
    manifest = fake_run(tmp_path)
    data = json.loads(manifest.read_text())
    del data["words"]
    data["clips"][0]["subtitles"] = "src_clip01.ass"
    data["clips"][0]["srt"] = "src_clip01.srt"
    manifest.write_text(json.dumps(data))
    (tmp_path / "src_clip01.ass").write_text("old captions")
    (tmp_path / "src_clip01.srt").write_text("old captions")
    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", fake_render)

    entry = rerender_clip(manifest, 1, 5, 20, log=lambda _m: None)

    assert entry["subtitles"] is None and entry["srt"] is None
    assert not (tmp_path / "src_clip01.ass").exists() and not (tmp_path / "src_clip01.srt").exists()
    assert (tmp_path / "src_clip01.mp4").read_bytes() == b"new clip"


@pytest.mark.parametrize("contents", [None, "not json", '[["a"]]'])
def test_unreadable_transcript_renders_without_captions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        contents: str | None):
    manifest = fake_run(tmp_path)
    data = json.loads(manifest.read_text())
    data["words"] = "src_words.json"
    manifest.write_text(json.dumps(data))
    if contents is not None:
        (tmp_path / "src_words.json").write_text(contents)
    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", fake_render)
    logs: list[str] = []

    entry = rerender_clip(manifest, 1, 5, 20, log=logs.append)

    assert entry["subtitles"] is None
    assert any("without captions" in m for m in logs)


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_rerender_real_clip_with_captions(tmp_path: Path):
    src = tmp_path / "long.mp4"
    make_video(src, duration=60)
    out = tmp_path / "out"
    job = Job(source=src, out_dir=out, clips=Options(count=2, min_len=10, max_len=15),
              render=RenderOptions(encoder="libx264", layout="fit"), captions=False)
    manifest = run(job, log=lambda _m: None)
    # pretend whisper ran so the captions path gets exercised
    save_words([Word(12.0, 12.5, "nudged"), Word(12.6, 13.0, "clip")], out / "long_words.json")
    data = json.loads(manifest.read_text())
    data["words"] = "long_words.json"
    manifest.write_text(json.dumps(data))

    entry = rerender_clip(manifest, 2, 11.0, 23.0, log=lambda _m: None)

    assert (entry["start"], entry["end"], entry["edited"]) == (11.0, 23.0, True)
    assert json.loads(manifest.read_text())["clips"][1] == entry
    clip = probe(out / entry["file"])
    assert (clip.width, clip.height) == (1080, 1920)
    assert abs(clip.duration - 12.0) < 0.2
    assert "NUDGED" in (out / entry["subtitles"]).read_text(encoding="utf-8")
    # the upload .srt is rebuilt for the new range too, relative to its start
    assert (out / entry["srt"]).read_text(encoding="utf-8") == "1\n00:00:01,000 --> 00:00:02,000\nnudged clip\n"
    assert not list(out.glob("*.tmp*"))


def test_moved_source_is_a_clear_error(tmp_path: Path):
    manifest = fake_run(tmp_path)
    (tmp_path / "src.mp4").unlink()
    with pytest.raises(ValueError, match="original video is gone"):
        rerender_clip(manifest, 1, 5, 20, log=lambda _m: None)
    assert (tmp_path / "src_clip01.mp4").read_bytes() == b"old clip"


def test_mark_clip(tmp_path: Path):
    manifest = fake_run(tmp_path)
    assert pipeline.mark_clip(manifest, 2, True)["keep"] is True
    assert pipeline.mark_clip(manifest, 2, None)["keep"] is None
    assert [c.get("keep", "unset") for c in json.loads(manifest.read_text())["clips"]] == ["unset", None]
    with pytest.raises(ValueError, match="doesn't exist"):
        pipeline.mark_clip(manifest, 3, True)


def test_no_srt_when_the_clip_has_no_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # words exist in the video, just not in the new range: captions file yes (empty), upload .srt no
    manifest = fake_run(tmp_path)
    data = json.loads(manifest.read_text())
    data["words"] = "src_words.json"
    data["clips"][0]["srt"] = "src_clip01.srt"
    manifest.write_text(json.dumps(data))
    save_words([Word(50.0, 50.5, "later")], tmp_path / "src_words.json")
    (tmp_path / "src_clip01.srt").write_text("old")
    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", fake_render)

    entry = rerender_clip(manifest, 1, 5, 20, log=lambda _m: None)

    assert entry["srt"] is None
    assert not (tmp_path / "src_clip01.srt").exists()
    assert not list(tmp_path.glob("*.tmp*"))


def test_failed_cover_is_one_short_log_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = fake_run(tmp_path)
    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", fake_render)
    logs: list[str] = []

    entry = rerender_clip(manifest, 1, 5, 20, log=logs.append)  # fake clip bytes: ffprobe can't read it

    assert entry["thumbnail"] is None
    assert any(m.endswith("no thumbnail for src_clip01.tmp.mp4 (ffprobe couldn't read it)") for m in logs)
    assert all(len(m) < 120 for m in logs)


def _capture_layouts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    layouts: list[str] = []

    def render(_info, _a, _b, out: Path, opts: RenderOptions, *_args, **_kw):
        layouts.append(opts.layout)
        out.write_bytes(b"new clip")

    monkeypatch.setattr(pipeline, "probe", lambda p: None)
    monkeypatch.setattr(pipeline, "render", render)
    return layouts


def test_layout_override_applies_to_one_clip_and_sticks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = fake_run(tmp_path)  # the run's layout is "fit"
    layouts = _capture_layouts(monkeypatch)
    lines: list[str] = []
    entry = rerender_clip(manifest, 1, 10, 15, log=lines.append, layout="crop")
    assert entry["layout"] == "crop" and "crop layout" in lines[0]
    # a later trim without a layout keeps the clip's own one
    rerender_clip(manifest, 1, 11, 15, log=lambda _m: None)
    # the other clip still uses the run's layout
    rerender_clip(manifest, 2, 20, 25, log=lambda _m: None)
    assert layouts == ["crop", "crop", "fit"]
    data = json.loads(manifest.read_text())
    assert data["clips"][0]["layout"] == "crop" and "layout" not in data["clips"][1]
    assert data["render"]["layout"] == "fit"


def test_bad_layout_is_refused_before_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = fake_run(tmp_path)
    before = manifest.read_text()
    monkeypatch.setattr(pipeline, "render", lambda *a, **k: pytest.fail("should not render"))
    with pytest.raises(ValueError, match="layout must be one of blur, crop, follow, speaker, fit"):
        rerender_clip(manifest, 1, 10, 15, log=lambda _m: None, layout="zoom")
    monkeypatch.setattr(pipeline.speaker, "available", lambda: False)
    with pytest.raises(ValueError, match="needs OpenCV"):
        rerender_clip(manifest, 1, 10, 15, log=lambda _m: None, layout="speaker")
    assert manifest.read_text() == before
