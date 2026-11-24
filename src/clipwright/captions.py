"""Word-by-word captions as an ASS subtitle file (burned in by ffmpeg/libass).

Words are grouped into short lines (a few words each, the usual shorts style) and the
word currently being spoken is highlighted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .transcribe import Word


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial Black"
    size: int = 84
    color: str = "FFFFFF"       # RRGGBB
    highlight: str = "FFD84D"
    outline: int = 6
    max_words: int = 3
    max_chars: int = 18
    margin_v: int = 520         # distance from the bottom on a 1920-tall frame


def _ass_color(rgb: str) -> str:
    rgb = rgb.lstrip("#")
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}".upper()  # ASS is AABBGGRR


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: Sequence[Word], max_words: int, max_chars: int, max_gap: float = 0.6) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        too_long = len(" ".join(x.text for x in [*cur, w])) > max_chars
        pause = bool(cur) and w.start - cur[-1].end > max_gap
        if cur and (len(cur) >= max_words or too_long or pause):
            groups.append(cur)
            cur = []
        cur.append(w)
        if w.text.endswith((".", "!", "?")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: Sequence[Word], clip_start: float, clip_end: float, style: CaptionStyle = CaptionStyle(),
              width: int = 1080, height: int = 1920, hold_gap: float = 0.6) -> str:
    """ASS document for the words inside [clip_start, clip_end], with times relative to the clip."""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{style.font},{style.size},{_ass_color(style.color)},{_ass_color(style.color)},&H00000000,"
        f"&H64000000,-1,0,0,0,100,100,0,0,1,{style.outline},2,2,60,60,{style.margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    hi = _ass_color(style.highlight)
    lines: list[str] = []
    groups = group_words(inside, style.max_words, style.max_chars, max_gap=hold_gap)
    for g, group in enumerate(groups):
        next_group_start = groups[g + 1][0].start if g + 1 < len(groups) else None
        for i, active in enumerate(group):
            start = max(active.start, clip_start) - clip_start
            # stay on screen until the next word starts so text doesn't flicker between words,
            # and between lines too unless there's a real pause
            if i + 1 < len(group):
                nxt = group[i + 1].start
            elif next_group_start is not None and next_group_start - active.end < hold_gap:
                nxt = next_group_start
            else:
                nxt = active.end
            end = min(max(nxt, active.end), clip_end) - clip_start
            if end <= start:
                continue
            parts = [
                f"{{\\c{hi}}}{_escape(w.text.upper())}{{\\r}}" if w is active else _escape(w.text.upper())
                for w in group
            ]
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + ("\n" if lines else "")


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(words: Sequence[Word], clip_start: float, clip_end: float, max_chars: int = 42,
              max_gap: float = 0.6) -> str:
    """Plain .srt for the words inside [clip_start, clip_end], times relative to the clip.

    This is the file you upload next to the video (YouTube, TikTok and others take one), so it
    uses normal subtitle lines up to max_chars (42 is the usual broadcast limit) instead of the
    3-word groups of the burned-in captions."""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    length = clip_end - clip_start
    blocks: list[str] = []
    for group in group_words(inside, max_words=max_chars, max_chars=max_chars, max_gap=max_gap):
        start = max(group[0].start, clip_start) - clip_start
        end = min(group[-1].end, clip_end) - clip_start
        if end - start < 0.05:
            continue
        text = " ".join(w.text for w in group)
        blocks.append(f"{len(blocks) + 1}\n{_srt_ts(start)} --> {_srt_ts(min(end, length))}\n{text}\n")
    return "\n".join(blocks)
