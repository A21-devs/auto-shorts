"""Words from an existing .srt file, so videos that already have subtitles skip Whisper.

SRT cues only have a start and end for the whole line, not per word. Each cue's time is
spread over its words in proportion to their length (a long word takes longer to say than
"a"), which is close enough for the word highlight and for snapping cut points to sentence
ends.
"""

from __future__ import annotations

import re
from pathlib import Path

from .transcribe import Word

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_CUE = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_TAG = re.compile(r"<[^>]*>|\{[^}]*\}")  # <i>, <font ...>, and {\an8} style overrides


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def split_cue(start: float, end: float, text: str) -> list[Word]:
    tokens = _TAG.sub("", text).split()
    if not tokens or end <= start:
        return []
    weights = [len(t) + 1 for t in tokens]  # +1 so short words still get some time
    total = sum(weights)
    words: list[Word] = []
    t = start
    for token, weight in zip(tokens, weights, strict=True):
        nxt = t + (end - start) * weight / total
        words.append(Word(round(t, 3), round(nxt, 3), token))
        t = nxt
    return words


def _norm(line: str) -> str:
    return " ".join(_TAG.sub("", line).split()).lower()


def _drop_repeated(previous: list[str], body: list[str]) -> list[str]:
    """Lines of this cue minus the ones carried over from the previous cue.

    YouTube auto-captions roll: each cue repeats the last line(s) of the one before and adds a
    new line. Without this every word would be captioned (and counted as speech) twice."""
    prev = [_norm(t) for t in previous]
    cur = [_norm(t) for t in body]
    for k in range(min(len(prev), len(cur)), 0, -1):
        if prev[-k:] == cur[:k]:
            return body[k:]
    return body


def parse_srt(text: str) -> list[Word]:
    """Words from SRT text. Blocks without a valid timing line are skipped, not fatal."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    words: list[Word] = []
    previous: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _CUE.search(line)
            if m:
                g = m.groups()
                body = [t for t in lines[i + 1:] if _TAG.sub("", t).strip()]
                new = _drop_repeated(previous, body)
                words += split_cue(_seconds(*g[:4]), _seconds(*g[4:]), " ".join(new))
                previous = body
                break
    words.sort(key=lambda w: w.start)
    return words


def load_srt(path: Path) -> list[Word]:
    # utf-8 first, then the Windows codepage a lot of older .srt files are saved in
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return parse_srt(text)


def sidecar(video: Path) -> Path | None:
    """`talk.srt` next to `talk.mp4`, if there is one."""
    for candidate in (video.with_suffix(".srt"), video.with_suffix(".SRT")):
        if candidate.is_file():
            return candidate
    return None
