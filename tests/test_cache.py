import json
import os
from pathlib import Path

import pytest

from clipwright.cache import AnalysisCache, cache_path
from clipwright.transcribe import Word


@pytest.fixture
def video(tmp_path: Path) -> Path:
    v = tmp_path / "game.mp4"
    v.write_bytes(b"not really a video")
    return v


def test_path_sits_next_to_the_video(video: Path):
    assert cache_path(video) == video.with_name("game.mp4.clipwright.json")


def test_round_trip(video: Path):
    c = AnalysisCache(video)
    assert c.get_audio(0.5) is None and c.get_cuts(0.35, 5) is None and c.get_words("small", None) is None
    c.put_audio(0.5, [-20.123, -50.0], silent=False)
    c.put_cuts(0.35, 5, [1.5, 20.25])
    c.put_words("small", None, [Word(1.0, 1.4, "héllo")])
    assert c.save()

    again = AnalysisCache(video)
    assert again.get_audio(0.5) == ([-20.12, -50.0], False)
    assert again.get_cuts(0.35, 5) == [1.5, 20.25]
    assert again.get_words("small", None) == [Word(1.0, 1.4, "héllo")]


def test_sections_are_keyed_by_their_settings(video: Path):
    c = AnalysisCache(video)
    c.put_audio(0.5, [-20.0], False)
    c.put_cuts(0.35, 5, [3.0])
    c.put_words("small", "en", [Word(0, 1, "hi")])
    assert c.get_audio(0.25) is None
    assert c.get_cuts(0.4, 5) is None and c.get_cuts(0.35, 10) is None
    assert c.get_words("small", None) is None and c.get_words("tiny", "en") is None
    # a second model is kept alongside the first
    c.put_words("tiny", None, [Word(0, 1, "yo")])
    assert c.get_words("small", "en") == [Word(0, 1, "hi")]


def test_changed_video_invalidates_everything(video: Path):
    c = AnalysisCache(video)
    c.put_cuts(0.35, 5, [3.0])
    c.save()
    video.write_bytes(b"a different video, longer")
    assert AnalysisCache(video).get_cuts(0.35, 5) is None


def test_touched_video_invalidates(video: Path):
    c = AnalysisCache(video)
    c.put_cuts(0.35, 5, [3.0])
    c.save()
    st = video.stat()
    os.utime(video, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
    assert AnalysisCache(video).get_cuts(0.35, 5) is None


@pytest.mark.parametrize("content", ["{broken", "[]", '{"version": 999}', ""])
def test_unreadable_cache_is_ignored(video: Path, content: str):
    cache_path(video).write_text(content, encoding="utf-8")
    c = AnalysisCache(video)
    assert c.get_audio(0.5) is None
    c.put_audio(0.5, [-1.0], True)
    assert c.save()
    assert json.loads(cache_path(video).read_text(encoding="utf-8"))["audio"]["silent"] is True


def test_damaged_sections_are_treated_as_missing(video: Path):
    c = AnalysisCache(video)
    c.data.update(audio={"hop": 0.5, "db": "nope"}, cuts={"key": "0.35@5", "times": [None]},
                  transcripts={"small|auto": [[1, 2]]})
    assert c.get_audio(0.5) is None and c.get_cuts(0.35, 5) is None and c.get_words("small", None) is None


def test_save_without_changes_writes_nothing(video: Path):
    assert AnalysisCache(video).save()
    assert not cache_path(video).exists()


def test_unwritable_location_is_not_an_error(video: Path, tmp_path: Path):
    c = AnalysisCache(video, path=tmp_path / "missing-dir" / "x.json")
    c.put_cuts(0.35, 5, [1.0])
    assert c.save() is False
    assert not (tmp_path / "missing-dir").exists()


def test_two_runs_saving_keep_each_others_transcripts(video: Path):
    a, b = AnalysisCache(video), AnalysisCache(video)   # both loaded before either saved
    a.put_words("tiny", None, [Word(0, 1, "tiny")])
    b.put_words("small", None, [Word(0, 1, "small")])
    b.put_cuts(0.35, 5, [2.0])
    assert a.save() and b.save()
    c = AnalysisCache(video)
    assert c.get_words("tiny", None) == [Word(0, 1, "tiny")]
    assert c.get_words("small", None) == [Word(0, 1, "small")]
    assert c.get_cuts(0.35, 5) == [2.0]


def test_save_does_not_bring_back_a_stale_cache_for_a_changed_video(video: Path):
    c = AnalysisCache(video)
    c.put_cuts(0.35, 5, [1.0])
    c.save()
    video.write_bytes(b"replaced with a different file")
    fresh = AnalysisCache(video)
    fresh.put_audio(0.5, [-10.0], False)
    fresh.save()
    again = AnalysisCache(video)
    assert again.get_cuts(0.35, 5) is None and again.get_audio(0.5) == ([-10.0], False)


def test_no_temp_files_left_behind(video: Path):
    c = AnalysisCache(video)
    c.put_cuts(0.35, 5, [1.0])
    c.save()
    assert sorted(p.name for p in video.parent.iterdir()) == ["game.mp4", "game.mp4.clipwright.json"]


def test_reads_a_cache_saved_under_the_old_name(video: Path):
    c = AnalysisCache(video)
    c.put_cuts(0.35, 5, [4.0])
    assert c.save()
    cache_path(video).rename(video.with_name("game.mp4.autoshorts.json"))

    again = AnalysisCache(video)
    assert again.get_cuts(0.35, 5) == [4.0]
    again.put_audio(0.5, [-20.0], False)
    assert again.save()
    # written back under the new name, with the old sections kept
    assert json.loads(cache_path(video).read_text(encoding="utf-8"))["cuts"]
    assert not video.with_name("game.mp4.autoshorts.json").exists()
