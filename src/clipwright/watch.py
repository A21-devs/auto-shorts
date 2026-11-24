"""Watch a recordings folder and hand each finished recording to a job.

OBS (and most recorders) write the file while recording, so a file only counts as finished once
its size and modification time have stayed the same for `settle` seconds. Partial files from
downloaders and editors (`.part`, `.tmp`, ...) are ignored, and so are recordings that already
have a manifest from an earlier run, so restarting the watcher doesn't redo anything.

Only the folder itself is scanned, not subfolders, so the `<name>_shorts` output folders that
land next to the recordings are never picked up as input.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = frozenset({".mp4", ".mkv", ".mov", ".flv", ".ts", ".webm", ".m4v", ".avi"})
PARTIAL_EXTS = frozenset({".part", ".tmp", ".crdownload", ".download", ".ytdl"})


def is_candidate(path: Path) -> bool:
    """A video file that isn't hidden, a temp file or a partial download (`x.mp4.part` has the
    suffix `.part`, `x.part.mp4` from some tools has `.part` in its suffixes)."""
    if path.name.startswith((".", "~")):
        return False
    suffixes = [s.lower() for s in path.suffixes]
    if not suffixes or suffixes[-1] not in VIDEO_EXTS:
        return False
    return not any(s in PARTIAL_EXTS for s in suffixes)


def out_dir_for(video: Path, out_root: Path | None) -> Path:
    """Where a recording's clips go: `<name>_shorts` next to it (like a normal run), or inside
    `out_root` when one is given."""
    return (out_root or video.parent) / f"{video.stem}_shorts"


def already_done(video: Path, out_dir: Path) -> bool:
    """True when `out_dir` has a manifest for this very file. The manifest is written last, so it
    only exists for runs that finished."""
    try:
        data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    source = data.get("source") if isinstance(data, dict) else None
    if not isinstance(source, str):
        return False
    try:
        return Path(source).resolve() == video.resolve()
    except OSError:
        return False


@dataclass
class _Seen:
    size: int
    mtime_ns: int
    since: float  # when it was first seen with this size and mtime


class Watcher:
    """Polls a folder and reports recordings that have stopped changing.

    `poll(now)` returns each finished recording once. A file that changes again after being
    reported (the recorder appended to it) is reported again once it settles."""

    def __init__(self, folder: Path, settle: float = 10.0) -> None:
        self.folder = folder
        self.settle = settle
        self._seen: dict[Path, _Seen] = {}
        self._reported: dict[Path, tuple[int, int]] = {}

    def poll(self, now: float) -> list[Path]:
        ready = []
        present = set()
        for path in sorted(self.folder.iterdir()):
            if not is_candidate(path):
                continue
            try:
                st = path.stat()
            except OSError:
                continue  # deleted or renamed between iterdir and stat
            if not path.is_file():
                continue
            present.add(path)
            key = (st.st_size, st.st_mtime_ns)
            seen = self._seen.get(path)
            if seen is None or (seen.size, seen.mtime_ns) != key:
                self._seen[path] = _Seen(st.st_size, st.st_mtime_ns, now)
                continue
            if st.st_size == 0 or now - seen.since < self.settle:
                continue
            if self._reported.get(path) != key:
                self._reported[path] = key
                ready.append(path)
        for gone in set(self._seen) - present:
            del self._seen[gone]
            self._reported.pop(gone, None)
        return ready


def watch(folder: Path, handle: Callable[[Path], None], settle: float = 10.0, interval: float = 2.0,
          out_root: Path | None = None, log: Callable[[str], None] = print,
          stop: Callable[[], bool] = lambda: False, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> None:
    """Calls `handle(video)` for every finished recording in `folder` that has no manifest yet,
    one at a time, until `stop()` returns True. A failing recording is logged and skipped; it's
    tried again only if the file changes."""
    watcher = Watcher(folder, settle)
    while not stop():
        for video in watcher.poll(clock()):
            if already_done(video, out_dir_for(video, out_root)):
                continue
            log(f"new recording: {video.name}")
            try:
                handle(video)
            except Exception as e:  # one bad file must not stop the watcher
                log(f"failed: {video.name}: {e}")
        sleep(interval)
