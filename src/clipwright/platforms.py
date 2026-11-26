"""Clip length presets per platform, so nobody has to remember each one's limits.

`limit` is the platform's hard maximum. `min_len`/`max_len` are the defaults clipwright picks
clips with: a bit under the limit, because jump cuts and re-renders only ever make a clip
shorter, and an upload that is 0.1s too long gets rejected (Shorts) or posted as a normal
video.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Platform:
    name: str
    min_len: float
    max_len: float
    limit: float
    aspect: str = "9:16"  # the frame clips get unless --aspect says otherwise


PLATFORMS: dict[str, Platform] = {
    "shorts": Platform("YouTube Shorts", 15, 58, 60),
    "tiktok": Platform("TikTok", 15, 60, 180),  # can go to 3 min, but 15-60s is what does well
    "reels": Platform("Instagram Reels", 15, 60, 90),
    # landscape uploads cut from a longer video: highlights for a second channel, a podcast clip
    "youtube": Platform("YouTube", 60, 300, 12 * 3600, "16:9"),
    "x": Platform("X", 15, 120, 140, "16:9"),  # 2:20 is the limit without Premium
    "linkedin": Platform("LinkedIn", 15, 90, 600, "1:1"),
}

DEFAULT_MIN = 20.0
DEFAULT_MAX = 55.0


def aspect_for(platform: str | None, aspect: str | None) -> str:
    """--aspect if it was given, else the platform's frame, else 9:16."""
    if aspect:
        return aspect
    p = PLATFORMS.get(platform or "")
    return p.aspect if p else "9:16"


def lengths(platform: str | None, min_len: float | None, max_len: float | None) -> tuple[float, float]:
    """Clip length range from an optional preset plus optional explicit values (which win).

    Raises ValueError for an unknown platform, or a --max over the platform's hard limit."""
    if platform is None:
        return (DEFAULT_MIN if min_len is None else min_len, DEFAULT_MAX if max_len is None else max_len)
    p = PLATFORMS.get(platform)
    if p is None:
        raise ValueError(f"unknown platform {platform!r}, pick one of {', '.join(PLATFORMS)}")
    lo = p.min_len if min_len is None else min_len
    hi = p.max_len if max_len is None else max_len
    if hi > p.limit:
        raise ValueError(f"{p.name} clips can be at most {p.limit:g}s, --max {hi:g} is over that")
    if max_len is None and lo > hi:
        hi = min(lo, p.limit)  # only --min given and it's above the preset's max: stretch the max to fit
    return lo, hi
