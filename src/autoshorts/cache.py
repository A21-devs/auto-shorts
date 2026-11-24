"""Saved analysis next to the video, so re-runs skip decoding and transcription.

`<video name>.autoshorts.json` holds the parts of the analysis that only depend on the video
and a few settings:

- audio: the loudness curve for a given hop (and whether the track was silent)
- cuts: scene cut times for a given threshold and fps
- transcripts: whisper words per (model, language)

It's tied to the video by size + modification time. If either changes, the whole file is
ignored. Each section has its own key, so turning captions on later reuses the audio and cuts
and only runs whisper.

A cache that can't be read is ignored, and one that can't be written (read-only folder, full
disk) is skipped. It's an optimisation, never a reason to fail a run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .transcribe import Word

VERSION = 1


def cache_path(source: Path) -> Path:
    return source.with_name(f"{source.name}.autoshorts.json")


def fingerprint(source: Path) -> dict[str, int]:
    st = source.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


class AnalysisCache:
    def __init__(self, source: Path, path: Path | None = None) -> None:
        self.source = source
        self.path = path or cache_path(source)
        self.dirty = False
        self.data: dict[str, Any] = {"version": VERSION, "source": fingerprint(source)}
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(saved, dict) and saved.get("version") == VERSION and saved.get("source") == self.data["source"]:
            self.data = saved

    # --- audio -------------------------------------------------------------------------------

    def get_audio(self, hop: float) -> tuple[list[float], bool] | None:
        a = self.data.get("audio")
        if not isinstance(a, dict) or a.get("hop") != hop:
            return None
        try:
            return [float(v) for v in a["db"]], bool(a["silent"])
        except (KeyError, TypeError, ValueError):
            return None

    def put_audio(self, hop: float, db: list[float], silent: bool) -> None:
        self.data["audio"] = {"hop": hop, "db": [round(float(v), 2) for v in db], "silent": silent}
        self.dirty = True

    # --- scene cuts --------------------------------------------------------------------------

    @staticmethod
    def _cuts_key(threshold: float, fps: int) -> str:
        return f"{threshold:g}@{fps}"

    def get_cuts(self, threshold: float, fps: int) -> list[float] | None:
        c = self.data.get("cuts")
        if not isinstance(c, dict) or c.get("key") != self._cuts_key(threshold, fps):
            return None
        try:
            return [float(t) for t in c["times"]]
        except (KeyError, TypeError, ValueError):
            return None

    def put_cuts(self, threshold: float, fps: int, times: list[float]) -> None:
        self.data["cuts"] = {"key": self._cuts_key(threshold, fps), "times": list(times)}
        self.dirty = True

    # --- transcripts -------------------------------------------------------------------------

    @staticmethod
    def _words_key(model: str, language: str | None) -> str:
        return f"{model}|{language or 'auto'}"

    def get_words(self, model: str, language: str | None) -> list[Word] | None:
        t = self.data.get("transcripts")
        if not isinstance(t, dict):
            return None
        entry = t.get(self._words_key(model, language))
        if entry is None:
            return None
        try:
            return [Word(float(a), float(b), str(text)) for a, b, text in entry]
        except (TypeError, ValueError):
            return None

    def put_words(self, model: str, language: str | None, words: list[Word]) -> None:
        t = self.data.setdefault("transcripts", {})
        if not isinstance(t, dict):
            t = self.data["transcripts"] = {}
        t[self._words_key(model, language)] = [[w.start, w.end, w.text] for w in words]
        self.dirty = True

    # --- disk --------------------------------------------------------------------------------

    def save(self) -> bool:
        """Writes the cache if anything changed. Returns False when it couldn't be written."""
        if not self.dirty:
            return True
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            tmp.unlink(missing_ok=True)
            return False
        self.dirty = False
        return True
