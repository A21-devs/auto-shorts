from pathlib import Path

from clipwright.subtitles import load_srt, parse_srt, sidecar, split_cue
from clipwright.transcribe import Word

SRT = """﻿1
00:00:01,000 --> 00:00:03,000
<i>Hello</i> there.

2
00:00:04,500 --> 00:00:05,500
{\\an8}Second
line here

garbage block without timing

3
01:00:00.250 --> 01:00:01.250
late
"""


def test_parses_cues_into_words():
    words = parse_srt(SRT.replace("\n", "\r\n"))
    assert [w.text for w in words] == ["Hello", "there.", "Second", "line", "here", "late"]
    assert words[0].start == 1.0 and words[1].end == 3.0
    assert words[2].start == 4.5 and words[4].end == 5.5
    assert words[-1] == Word(3600.25, 3601.25, "late")


def test_longer_words_get_more_time():
    a, b = split_cue(0.0, 3.0, "a extraordinary")
    assert b.end - b.start > 4 * (a.end - a.start)
    assert a.end == b.start


def test_empty_and_backwards_cues_are_skipped():
    assert split_cue(5.0, 5.0, "x") == []
    assert split_cue(5.0, 6.0, "<b></b>") == []


def test_cp1252_file(tmp_path: Path):
    f = tmp_path / "old.srt"
    f.write_bytes("1\n00:00:00,000 --> 00:00:01,000\ncaf\xe9\n".encode("cp1252"))
    assert [w.text for w in load_srt(f)] == ["caf\xe9"]


def test_sidecar(tmp_path: Path):
    video = tmp_path / "talk.mp4"
    assert sidecar(video) is None
    (tmp_path / "talk.srt").write_text("")
    assert sidecar(video) == tmp_path / "talk.srt"
