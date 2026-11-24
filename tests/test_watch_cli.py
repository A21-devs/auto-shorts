import shutil
from pathlib import Path

import pytest
from conftest import make_video

from clipwright import watch
from clipwright.cli import main


def test_watch_needs_a_folder(tmp_path: Path, capsys):
    assert main(["watch", str(tmp_path / "nope")]) == 2
    assert "is not a folder" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--settle", "-1"], ["--interval", "0"], ["-n", "0"], ["--min", "30", "--max", "10"],
                                  ["--copy", "ollama", "--no-captions"]])
def test_watch_checks_options_before_starting(tmp_path: Path, args: list[str], capsys,
                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(watch, "watch", lambda *a, **k: pytest.fail("should not start"))
    assert main(["watch", str(tmp_path), *args]) == 2
    assert "error:" in capsys.readouterr().err


def test_watch_needs_ffmpeg_up_front(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(watch, "watch", lambda *a, **k: pytest.fail("should not start"))
    assert main(["watch", str(tmp_path)]) == 3
    assert "not found on PATH" in capsys.readouterr().err


def test_watch_stops_cleanly_on_ctrl_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    def interrupted(*a: object, **k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(watch, "watch", interrupted)
    assert main(["watch", str(tmp_path)]) == 0
    assert "stopped" in capsys.readouterr().out


@pytest.mark.ffmpeg
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("out", [False, True])
def test_watch_handler_writes_where_already_done_looks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, out: bool):
    # runs the real handler on a real video, then checks the watcher would skip it next time
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    video = rec_dir / "stream.mp4"
    make_video(video, duration=30)
    out_root = tmp_path / "clips" if out else None
    calls: list[Path] = []

    def once(folder: Path, handle, settle: float, interval: float, out_root: Path | None) -> None:
        assert folder == rec_dir
        handle(video)
        calls.append(video)
        assert watch.already_done(video, watch.out_dir_for(video, out_root))

    monkeypatch.setattr(watch, "watch", once)
    args = ["watch", str(rec_dir), "-n", "1", "--min", "5", "--max", "10", "--no-captions", "--encoder", "libx264"]
    if out_root:
        args += ["-o", str(out_root)]
    assert main(args) == 0
    assert calls == [video]
    expected = (out_root or rec_dir) / "stream_shorts" / "stream_clip01.mp4"
    assert expected.is_file()
