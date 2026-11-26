"""--caption-style presets and the --hook title."""

import json
import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright.captions import HOOK_MAX_CHARS, STYLES, build_ass, hook_text, wrap
from clipwright.pipeline import Job, rerender_clip, run
from clipwright.render import RenderOptions
from clipwright.segment import Options
from clipwright.server import BadRequest, parse_job
from clipwright.transcribe import Word

WORDS = [Word(0.2, 0.6, "Wait"), Word(0.6, 1.0, "for"), Word(1.0, 1.4, "it."), Word(2.0, 2.4, "Okay")]


def style_line(doc: str, name: str) -> list[str]:
    line = next(x for x in doc.splitlines() if x.startswith(f"Style: {name},"))
    return line.split(":", 1)[1].split(",")


def test_bold_is_the_old_default():
    doc = build_ass(WORDS, 0, 5, STYLES["bold"])
    assert "WAIT" in doc and "Wait" not in doc.split("[Events]")[1]
    cap = style_line(doc, "Cap")
    assert cap[1] == "Arial Black" and cap[7] == "-1" and cap[15] == "1"  # bold, outline border


def test_clean_keeps_case_and_boxed_uses_a_box():
    assert "Wait" in build_ass(WORDS, 0, 5, STYLES["clean"]).split("[Events]")[1]
    boxed = style_line(build_ass(WORDS, 0, 5, STYLES["boxed"]), "Cap")
    assert boxed[15] == "3"  # opaque box


def test_minimal_is_not_bold_and_pop_is_two_words():
    assert style_line(build_ass(WORDS, 0, 5, STYLES["minimal"]), "Cap")[7] == "0"
    pop = build_ass(WORDS, 0, 5, STYLES["pop"]).split("[Events]")[1]
    assert all(line.count(" ") <= 9 for line in pop.splitlines() if line.startswith("Dialogue"))


def test_every_style_renders_every_frame():
    for name, style in STYLES.items():
        for w, h in [(1080, 1920), (1920, 1080), (1080, 1080)]:
            doc = build_ass(WORDS, 0, 5, style.for_frame(w, h), width=w, height=h, hook="A hook")
            assert doc.count("Dialogue:") >= 2, name


def test_hook_is_on_top_for_three_seconds():
    doc = build_ass(WORDS, 0, 30, STYLES["bold"], hook="He finally beat the boss after two hours of trying")
    hook = next(x for x in doc.splitlines() if ",Hook," in x)
    assert hook.startswith("Dialogue: 1,0:00:00.00,0:00:03.00,Hook,")
    assert "\\N" in hook  # wrapped, libass won't wrap with WrapStyle 2
    assert style_line(doc, "Hook")[18] == "8"  # top centre


def test_hook_never_outlasts_a_short_clip():
    doc = build_ass([], 0, 2, STYLES["bold"], hook="Short")
    assert "0:00:00.00,0:00:02.00,Hook" in doc


def test_hook_text_is_the_first_sentence():
    assert hook_text(WORDS, 0, 10) == "Wait for it."
    long = [Word(i, i + 0.5, "word") for i in range(40)]
    text = hook_text(long, 0, 60)
    assert text is not None and text.endswith("...") and len(text) <= HOOK_MAX_CHARS + 3
    assert hook_text(WORDS, 3, 10) is None  # nothing said in the first half


def test_wrap():
    assert wrap("one two three four", 9) == ["one two", "three", "four"]


def test_server_validates_style_and_hook(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    job = parse_job({"video": str(video), "caption_style": "boxed", "hook": True})
    assert job.render.caption_style == "boxed" and job.render.hook
    with pytest.raises(BadRequest, match="caption_style"):
        parse_job({"video": str(video), "caption_style": "comic"})
    with pytest.raises(BadRequest, match="hook"):
        parse_job({"video": str(video), "hook": "yes"})


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_hook_is_burned_in_and_kept_on_rerender(tmp_path: Path):
    src = tmp_path / "long.mp4"
    make_video(src)
    srt = tmp_path / "long.srt"
    srt.write_text("\n".join(f"{i + 1}\n00:00:{i * 5:02d},000 --> 00:00:{i * 5 + 4:02d},000\nline number {i}.\n"
                             for i in range(12)), encoding="utf-8")
    job = Job(source=src, out_dir=tmp_path / "out", clips=Options(count=1, min_len=15, max_len=20),
              render=RenderOptions(encoder="libx264", caption_style="clean", hook=True), srt=srt)
    manifest_path = run(job, log=lambda _m: None)
    data = json.loads(manifest_path.read_text())
    clip = data["clips"][0]
    assert "line number" in clip["hook"]
    assert data["render"]["caption_style"] == "clean"
    ass = (tmp_path / "out" / clip["subtitles"]).read_text(encoding="utf-8")
    assert ",Hook," in ass and clip["hook"] in ass
    rerender_clip(manifest_path, 1, clip["start"] + 1, clip["end"], log=lambda _m: None)
    again = (tmp_path / "out" / clip["subtitles"]).read_text(encoding="utf-8")
    assert clip["hook"] in again
