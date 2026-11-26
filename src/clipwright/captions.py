"""Word-by-word captions as an ASS subtitle file (burned in by ffmpeg/libass).

Words are grouped into short lines (a few words each, the usual shorts style) and the
word currently being spoken is highlighted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

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
    uppercase: bool = True
    box: bool = False           # text on a dark box instead of an outline
    bold: bool = True

    def for_frame(self, width: int, height: int) -> CaptionStyle:
        """The same look on another frame size. Sizes are tuned for 1080x1920; a shorter frame gets
        smaller text (not below 65%, or it stops reading on a phone) and a proportional margin."""
        if (width, height) == (1080, 1920):
            return self
        scale = height / 1920
        wide = width > height
        text = max(scale, 0.65)
        return replace(self, size=round(self.size * text), outline=max(2, round(self.outline * text)),
                       margin_v=round(self.margin_v * scale * (0.5 if wide else 1)),
                       max_chars=self.max_chars * 2 if wide else self.max_chars,
                       max_words=self.max_words + 2 if wide else self.max_words)


# --caption-style presets. bold is the original look and stays the default.
STYLES: dict[str, CaptionStyle] = {
    "bold": CaptionStyle(),
    # mixed case in a plain heavy sans, green highlight: the podcast-clip look
    "clean": CaptionStyle(font="Arial", size=78, highlight="5BE37D", outline=5, max_words=4, max_chars=22,
                          uppercase=False),
    # white text on a dark box, for busy gameplay where an outline gets lost
    "boxed": CaptionStyle(font="Arial", size=72, highlight="FFD84D", outline=14, max_words=4, max_chars=24,
                          uppercase=False, box=True),
    # big, one or two words at a time, pink highlight: fast-cut energy
    "pop": CaptionStyle(size=104, highlight="FF4FA3", outline=8, max_words=2, max_chars=12),
    # small and low, no highlight: for clips where the picture matters more than the words
    "minimal": CaptionStyle(font="Arial", size=60, highlight="FFFFFF", outline=4, max_words=6, max_chars=32,
                            margin_v=260, uppercase=False, bold=False),
}

HOOK_SECONDS = 3.0
HOOK_MAX_CHARS = 60


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


def wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap (WrapStyle 2 means libass never wraps on its own)."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def hook_text(words: Sequence[Word], clip_start: float, clip_end: float) -> str | None:
    """A hook title from the clip's own words when there's no AI title: its first sentence, cut at
    HOOK_MAX_CHARS on a word boundary. None when nothing is said in the first half."""
    inside = [w for w in words if w.end > clip_start and w.start < clip_start + (clip_end - clip_start) / 2]
    out = ""
    for w in inside:
        nxt = f"{out} {w.text}".strip()
        if len(nxt) > HOOK_MAX_CHARS:
            out += "..."
            break
        out = nxt
        if w.text.endswith((".", "!", "?")) and len(out) >= 12:
            break
    return out or None


def build_ass(words: Sequence[Word], clip_start: float, clip_end: float, style: CaptionStyle = CaptionStyle(),
              width: int = 1080, height: int = 1920, hold_gap: float = 0.6, hook: str | None = None) -> str:
    """ASS document for the words inside [clip_start, clip_end], with times relative to the clip.
    `hook` is a title shown at the top for the first HOOK_SECONDS."""
    inside = [w for w in words if w.end > clip_start and w.start < clip_end]
    border = 3 if style.box else 1  # 3 = opaque box behind the text, sized by the outline
    back = "&H50000000" if style.box else "&H64000000"
    outline_colour = "&H50000000" if style.box else "&H00000000"
    hook_size = round(style.size * 0.8)
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{style.font},{style.size},{_ass_color(style.color)},{_ass_color(style.color)},{outline_colour},"
        f"{back},{-1 if style.bold else 0},0,0,0,100,100,0,0,{border},{style.outline},"
        f"{0 if style.box else 2},2,60,60,{style.margin_v},1\n"
        f"Style: Hook,{style.font},{hook_size},&H00FFFFFF,&H00FFFFFF,&H28000000,&H28000000,-1,0,0,0,100,100,0,0,3,"
        f"{max(8, hook_size // 5)},0,8,60,60,{round(height * 0.09)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    hi = _ass_color(style.highlight)
    lines: list[str] = []
    if hook and hook.strip():
        per_line = max(12, int((width - 160) / (hook_size * 0.6)))
        text = "\\N".join(_escape(line) for line in wrap(hook.strip(), per_line)[:3])
        until = min(HOOK_SECONDS, clip_end - clip_start)
        lines.append(f"Dialogue: 1,{_ts(0)},{_ts(until)},Hook,,0,0,0,,{{\\fad(120,250)}}{text}")
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
            def shown(w: Word) -> str:
                return _escape(w.text.upper() if style.uppercase else w.text)

            parts = [f"{{\\c{hi}}}{shown(w)}{{\\r}}" if w is active else shown(w) for w in group]
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
