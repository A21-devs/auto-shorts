import json
import os
from pathlib import Path

import pytest

from clipwright.watch import Watcher, already_done, is_candidate, out_dir_for, watch


@pytest.mark.parametrize(("name", "ok"), [
    ("stream.mp4", True), ("Stream 2026-11-24.MKV", True), ("a.flv", True), ("clip.ts", True),
    ("notes.txt", False), ("noext", False), (".hidden.mp4", False), ("~lock.mp4", False),
    ("video.mp4.part", False), ("video.part.mp4", False), ("video.mkv.tmp", False),
    ("video.mp4.crdownload", False),
])
def test_is_candidate(name: str, ok: bool):
    assert is_candidate(Path(name)) is ok


def test_out_dir_for(tmp_path: Path):
    video = tmp_path / "rec.mkv"
    assert out_dir_for(video, None) == tmp_path / "rec_shorts"
    assert out_dir_for(video, tmp_path / "out") == tmp_path / "out" / "rec_shorts"


def test_already_done_needs_a_manifest_for_this_file(tmp_path: Path):
    video = tmp_path / "rec.mp4"
    video.write_bytes(b"x")
    out = tmp_path / "rec_shorts"
    assert not already_done(video, out)
    out.mkdir()
    (out / "manifest.json").write_text("{broken", encoding="utf-8")
    assert not already_done(video, out)
    (out / "manifest.json").write_text(json.dumps({"source": str(tmp_path / "other.mp4")}), encoding="utf-8")
    assert not already_done(video, out)
    (out / "manifest.json").write_text(json.dumps({"source": str(video)}), encoding="utf-8")
    assert already_done(video, out)


def _bump(path: Path, data: bytes, mtime: int) -> None:
    with path.open("ab") as f:
        f.write(data)
    os.utime(path, ns=(mtime, mtime))


def test_watcher_waits_until_the_file_stops_growing(tmp_path: Path):
    w = Watcher(tmp_path, settle=10)
    rec = tmp_path / "rec.mp4"
    _bump(rec, b"a", 1_000)
    assert w.poll(0) == []           # first sighting
    _bump(rec, b"b", 2_000)
    assert w.poll(8) == []           # grew: the clock restarts
    assert w.poll(17) == []          # 9 s unchanged
    assert w.poll(18) == [rec]       # 10 s unchanged
    assert w.poll(30) == []          # reported once
    _bump(rec, b"c", 3_000)          # the recorder appended more
    assert w.poll(31) == []
    assert w.poll(41) == [rec]


def test_watcher_ignores_empty_partial_and_subfolders(tmp_path: Path):
    w = Watcher(tmp_path, settle=1)
    (tmp_path / "empty.mp4").touch()
    (tmp_path / "dl.mp4.part").write_bytes(b"x")
    (tmp_path / "rec_shorts").mkdir()
    (tmp_path / "rec_shorts" / "rec_clip01.mp4").write_bytes(b"x")
    (tmp_path / "folder.mp4").mkdir()
    assert w.poll(0) == []
    assert w.poll(5) == []


def test_watcher_forgets_deleted_files(tmp_path: Path):
    w = Watcher(tmp_path, settle=1)
    rec = tmp_path / "rec.mp4"
    _bump(rec, b"a", 1_000)
    w.poll(0)
    assert w.poll(2) == [rec]
    rec.unlink()
    assert w.poll(3) == []
    _bump(rec, b"a", 1_000)  # same name and stat again: a new recording, report it again
    w.poll(4)
    assert w.poll(6) == [rec]


def test_watch_runs_new_recordings_and_skips_done_ones(tmp_path: Path):
    done = tmp_path / "old.mp4"
    _bump(done, b"x", 1_000)
    (tmp_path / "old_shorts").mkdir()
    (tmp_path / "old_shorts" / "manifest.json").write_text(json.dumps({"source": str(done)}), encoding="utf-8")
    bad = tmp_path / "bad.mp4"
    _bump(bad, b"x", 1_000)
    new = tmp_path / "new.mkv"
    _bump(new, b"x", 1_000)

    handled: list[str] = []
    logs: list[str] = []
    now = [0.0]

    def handle(video: Path) -> None:
        handled.append(video.name)
        if video == bad:
            raise RuntimeError("ffmpeg failed")

    def sleep(s: float) -> None:
        now[0] += s

    watch(tmp_path, handle, settle=3, interval=2, log=logs.append,
          stop=lambda: now[0] > 20, clock=lambda: now[0], sleep=sleep)
    assert handled == ["bad.mp4", "new.mkv"]  # each once, in name order, old.mp4 skipped
    assert "failed: bad.mp4: ffmpeg failed" in logs
    assert "new recording: new.mkv" in logs


def test_watch_survives_the_folder_going_away(tmp_path: Path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    logs: list[str] = []
    handled: list[str] = []
    now = [0.0]

    def sleep(s: float) -> None:
        now[0] += s
        if now[0] == 2:
            folder.rename(tmp_path / "unplugged")
        if now[0] == 10:
            (tmp_path / "unplugged").rename(folder)
            _bump(folder / "rec.mp4", b"x", 1_000)

    watch(folder, lambda v: handled.append(v.name), settle=2, interval=1, log=logs.append,
          stop=lambda: now[0] > 20, clock=lambda: now[0], sleep=sleep)
    assert sum("waiting for it to come back" in m for m in logs) == 1  # said once, not every poll
    assert any(m.endswith("is back") for m in logs)
    assert handled == ["rec.mp4"]
