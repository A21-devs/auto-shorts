"""--pick: let a language model help choose which moments become clips.

The signal scoring (segment.py) is good at finding where something happens and bad at knowing
whether what's said makes sense on its own. So it proposes a shortlist of non-overlapping
windows, a model reads each window's transcript and rates it as a standalone short, and the two
are blended. The model never sees audio or video and can't invent ranges: it only rates the
numbered candidates it was given.

If the model can't be reached, or answers with junk, the run carries on with the signal pick.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from . import copywriting
from .segment import Clip, Options, Signals, candidates, pick
from .transcribe import Word

SHORTLIST_FACTOR = 3   # rate 3x as many windows as clips wanted
MAX_SHORTLIST = 24     # keeps one prompt under ~15k words even for 3 min clips
MAX_WORDS_EACH = 450
AI_WEIGHT = 0.75       # share of the final score that comes from the model: the shortlist is already
                       # the loudest moments, what's said is what separates them

SYSTEM = (
    "You pick moments from a long video to post as short clips. You get numbered candidate clips, each "
    "with its transcript. Rate every candidate from 0 to 10 for how well it works as a standalone short: "
    "it hooks in the first seconds, makes sense without the rest of the video, has a payoff (a punchline, "
    "a reveal, a strong opinion, a useful tip, a reaction), and doesn't stop mid-thought. Rate low if it's "
    "filler, small talk, an ad read, or needs context. Give a few words on why. Rate every candidate exactly "
    "once, using its number."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "score": {"type": "number"},
                    "why": {"type": "string"},
                },
                "required": ["n", "score", "why"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["ratings"],
    "additionalProperties": False,
}

Ask = Callable[[str, str, dict[str, Any]], Any]  # (prompt, system, schema) -> parsed JSON


def shortlist(sig: Signals, opt: Options) -> list[Clip]:
    """The best non-overlapping windows by signal score, more than will be kept."""
    size = min(MAX_SHORTLIST, max(opt.count, opt.count * SHORTLIST_FACTOR))
    return pick(candidates(sig, opt), replace(opt, count=size))


def transcript(words: Sequence[Word], clip: Clip) -> str:
    inside = [w.text for w in words if w.end > clip.start and w.start < clip.end]
    return " ".join(inside[:MAX_WORDS_EACH])


def prompt_for(clips: Sequence[Clip], words: Sequence[Word], platform: str | None) -> str:
    where = f" They will be posted on {platform}." if platform else ""
    parts = [f"{len(clips)} candidate clips.{where}"]
    for i, c in enumerate(clips, 1):
        parts.append(f"\n#{i} ({c.duration:.0f}s)\n{transcript(words, c) or '(nothing is said)'}")
    return "\n".join(parts)


def parse(data: Any, count: int) -> dict[int, tuple[float, str]]:
    """{candidate number: (score 0-10, why)} from the model's answer. Unknown numbers, repeats and
    non-numbers are dropped; scores are clamped to 0-10."""
    if not isinstance(data, dict) or not isinstance(data.get("ratings"), list):
        raise copywriting.CopyError("the answer had no ratings")
    out: dict[int, tuple[float, str]] = {}
    for r in data["ratings"]:
        if not isinstance(r, dict):
            continue
        n, score, why = r.get("n"), r.get("score"), r.get("why")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= count or n in out:
            continue
        if isinstance(score, bool) or not isinstance(score, int | float):
            continue
        out[n] = (min(10.0, max(0.0, float(score))), str(why or "").strip()[:80])
    if not out:
        raise copywriting.CopyError("the answer rated none of the candidates")
    return out


def blend(clips: Sequence[Clip], ratings: dict[int, tuple[float, str]], count: int) -> list[Clip]:
    """Final pick: signal score (scaled 0-1 over the shortlist) mixed with the model's rating.
    A candidate the model skipped keeps its signal score with a neutral 5/10. Returned in time order."""
    lo = min(c.score for c in clips)
    span = (max(c.score for c in clips) - lo) or 1.0
    scored: list[Clip] = []
    for i, c in enumerate(clips, 1):
        ai, why = ratings.get(i, (5.0, ""))
        final = (1 - AI_WEIGHT) * (c.score - lo) / span + AI_WEIGHT * ai / 10
        reasons = (*c.reasons, f"AI {ai:g}/10: {why}" if why else f"AI {ai:g}/10")
        scored.append(Clip(c.start, c.end, round(final * 10, 2), reasons))
    best = sorted(scored, key=lambda c: c.score, reverse=True)[:count]
    return sorted(best, key=lambda c: c.start)


def find_clips(sig: Signals, opt: Options, words: Sequence[Word], ask: Ask, platform: str | None = None,
               log: Callable[[str], None] = print) -> list[Clip]:
    """Clips picked with the model's help, or the plain signal pick if that isn't possible."""
    short = shortlist(sig, opt)
    if not words:
        log("no transcript, so --pick has nothing to read: picking from audio and scenes only")
        return pick(short, opt)
    if len(short) <= opt.count:
        return short  # nothing to choose between
    log(f"asking the model to rate {len(short)} candidates")
    try:
        ratings = parse(ask(prompt_for(short, words, platform), SYSTEM, SCHEMA), len(short))
    except copywriting.CopyError as e:
        log(f"AI pick failed ({e}), picking from audio and scenes only")
        return pick(short, opt)
    log(f"model rated {len(ratings)} of {len(short)}")
    return blend(short, ratings, opt.count)
