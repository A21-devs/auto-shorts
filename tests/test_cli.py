from pathlib import Path

import pytest

from autoshorts.cli import main
from autoshorts.pipeline import remove_stale_clips


@pytest.mark.parametrize("args", [["-n", "0"], ["-n", "-3"], ["--min", "0"], ["--min", "30", "--max", "20"]])
def test_bad_numbers_are_rejected(tmp_path: Path, args: list[str], capsys):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"not really a video")
    assert main([str(video), *args]) == 2
    assert "error:" in capsys.readouterr().err


def test_missing_file(tmp_path: Path):
    assert main([str(tmp_path / "nope.mp4")]) == 2


def test_stale_clips_removed_but_nothing_else(tmp_path: Path):
    for name in ["vod_clip01.mp4", "vod_clip01.ass", "vod_clip02.mp4", "vod_clip05.mp4", "vod_clip05.ass",
                 "vod_clip05_edit.mp4", "other_clip09.mp4", "notes.txt"]:
        (tmp_path / name).write_text("x")
    assert remove_stale_clips(tmp_path, "vod", keep=2) == 2
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == ["notes.txt", "other_clip09.mp4", "vod_clip01.ass", "vod_clip01.mp4", "vod_clip02.mp4",
                    "vod_clip05_edit.mp4"]
