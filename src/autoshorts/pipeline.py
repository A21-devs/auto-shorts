"""Long video in, folder of shorts + manifest.json out."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from . import audio, jumpcut, scenes, segment, transcribe
from .captions import CaptionStyle, build_ass
from .probe import MediaInfo, probe
from .render import RenderOptions, render

Log = Callable[[str], None]
# (stage, fraction of that stage done). Stages in order: audio, scenes, transcribe (only with
# captions), render, done. Fractions go from 0 to 1 and never go down within a stage.
Progress = Callable[[str, float], None]

MIN_CLIP = 1.0  # shortest range rerender_clip accepts, in seconds


@dataclass
class Job:
    source: Path
    out_dir: Path
    clips: segment.Options = field(default_factory=segment.Options)
    render: RenderOptions = field(default_factory=RenderOptions)
    captions: bool = True
    whisper_model: str = "small"
    language: str | None = None
    hop: float = 0.5


def _no_progress(_stage: str, _fraction: float) -> None:
    pass


def analyze(info: MediaInfo, job: Job, log: Log,
            progress: Progress = _no_progress) -> tuple[segment.Signals, list[transcribe.Word]]:
    t = time.perf_counter()
    pcm = None
    progress("audio", 0.0)
    if info.has_audio:
        log("reading audio")
        pcm = audio.decode_pcm(info.path)
        db = audio.loudness_curve(pcm, hop=job.hop)
        quiet = audio.silences(db, job.hop)
        if len(db) and float(np.max(db)) <= audio.SILENCE_DB + 1.0:
            # Roblox's recorder (and muted OBS tracks) write an audio stream of pure silence.
            # Treat it as no audio, otherwise every window ties and it just picks the start.
            log("audio track is silent, picking clips from scene changes only")
            pcm = None
            db = np.zeros(len(db))
            quiet = []
    else:
        db = np.zeros(int(info.duration / job.hop))
        quiet = []
    progress("audio", 1.0)
    log("finding scene cuts")
    progress("scenes", 0.0)
    cuts = scenes.detect_cuts(info.path, info.video_stream)
    progress("scenes", 1.0)

    words: list[transcribe.Word] = []
    if job.captions and pcm is not None:
        if transcribe.available():
            log(f"transcribing (whisper {job.whisper_model})")
            progress("transcribe", 0.0)
            words = transcribe.transcribe(pcm, job.whisper_model, language=job.language, log=log,
                                          on_progress=lambda f: progress("transcribe", f))
            progress("transcribe", 1.0)
        else:
            log("faster-whisper not installed, skipping captions (pip install autoshorts[transcribe])")

    boundaries = sorted({round((a + b) / 2, 2) for a, b in quiet} | set(transcribe.sentence_ends(words)))
    sig = segment.Signals(
        hop=job.hop,
        loudness=audio.zscore(db),
        speech=transcribe.speech_density(words, len(db) * job.hop, job.hop) if words else None,
        cuts=tuple(cuts),
        boundaries=tuple(boundaries),
    )
    log(f"analysis done in {time.perf_counter() - t:.1f}s ({len(cuts)} cuts, {len(words)} words)")
    return sig, words


def remove_stale_clips(out_dir: Path, stem: str, keep: int) -> int:
    """Deletes <stem>_clipNN.mp4/.ass with NN > keep. Only touches files that match our own naming."""
    pattern = re.compile(re.escape(stem) + r"_clip(\d{2,})\.(mp4|ass)")
    removed = 0
    for f in out_dir.iterdir():
        m = pattern.fullmatch(f.name)
        if m and int(m.group(1)) > keep:
            f.unlink()
            removed += 1
    return removed


def plan_cuts(info: MediaInfo, start: float, end: float) -> jumpcut.Ranges | None:
    """Clip-relative ranges to keep for --jumpcut, or None when there's nothing to cut."""
    if not info.has_audio:
        return None
    pcm = audio.decode_pcm(info.path, start=start, duration=end - start)
    keep = jumpcut.keep_ranges(end - start, jumpcut.dead_air(pcm))
    if not keep or keep == [(0.0, round(end - start, 3))]:
        return None  # nothing to cut, or nothing left (all silence): render it as it is
    return keep


def _render_one(info: MediaInfo, words: list[transcribe.Word], start: float, end: float, out_dir: Path,
                base: str, opts: RenderOptions, suffix: str = "") -> tuple[str, str | None, float]:
    """Writes <base><suffix>.ass (if there are words) and renders <base><suffix>.mp4.
    Returns both names and the length of the rendered clip (shorter than end - start with jump cuts)."""
    name = f"{base}{suffix}.mp4"
    keep = plan_cuts(info, start, end) if opts.jumpcut else None
    length = jumpcut.kept_duration(keep) if keep else end - start
    sub = None
    if words:
        sub = f"{base}{suffix}.ass"
        if keep:
            ass = build_ass(jumpcut.retime_words(words, start, keep), 0.0, length, CaptionStyle())
        else:
            ass = build_ass(words, start, end, CaptionStyle())
        (out_dir / sub).write_text(ass, encoding="utf-8")
    render(info, start, end, out_dir / name, opts, sub, workdir=out_dir, keep=keep)
    return name, sub, length


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    # write then rename, so a reader (the review page) never sees half a file
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(job: Job, log: Log = print, progress: Progress = _no_progress) -> Path:
    info = probe(job.source)
    log(f"{info.path.name}: {info.duration / 60:.1f} min, {info.width}x{info.height} @ {info.fps:.0f}fps")
    sig, words = analyze(info, job, log, progress)
    clips = segment.find_clips(sig, job.clips)
    if not clips:
        raise RuntimeError("no clips found, is the video shorter than --min?")

    out_dir = job.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    removed = remove_stale_clips(out_dir, stem, keep=len(clips))
    if removed:
        log(f"removed {removed} clip file(s) left over from an earlier run")
    words_file = None
    if words:
        words_file = f"{stem}_words.json"
        transcribe.save_words(words, out_dir / words_file)
    results = []
    progress("render", 0.0)
    for i, clip in enumerate(clips, 1):
        why = ", ".join(clip.reasons)
        log(f"[{i}/{len(clips)}] {clip.start:7.1f}s - {clip.end:7.1f}s  score {clip.score:5.2f}  {why}")
        t = time.perf_counter()
        name, sub, length = _render_one(info, words, clip.start, clip.end, out_dir, f"{stem}_clip{i:02d}",
                                        job.render)
        cut = clip.duration - length
        log(f"         rendered {name} in {time.perf_counter() - t:.1f}s"
            + (f", cut {cut:.1f}s of pauses ({length:.1f}s left)" if cut > 0.05 else ""))
        results.append({**asdict(clip), "file": name, "subtitles": sub, "length": round(length, 2)})
        progress("render", i / len(clips))

    manifest = out_dir / "manifest.json"
    _write_manifest(manifest, {"source": str(info.path), "duration": info.duration, "render": asdict(job.render),
                               "words": words_file, "clips": results})
    progress("done", 1.0)
    return manifest


def _render_options(saved: dict[str, Any] | None) -> RenderOptions:
    """RenderOptions from a manifest. Older manifests don't have them, unknown keys are ignored."""
    known = {f.name for f in fields(RenderOptions)}
    return RenderOptions(**{k: v for k, v in (saved or {}).items() if k in known})


def rerender_clip(manifest: Path, n: int, start: float, end: float, log: Log = print) -> dict[str, Any]:
    """Re-renders clip number n (1-based) of a finished run over [start, end] and updates manifest.json.

    Renders to a temp name first and swaps it in at the end, so a failed render leaves the old clip
    (and its captions) as they were.
    """
    data = json.loads(manifest.read_text(encoding="utf-8"))
    clips = data["clips"]
    if not 1 <= n <= len(clips):
        raise ValueError(f"clip {n} doesn't exist, this run has {len(clips)}")
    start, end = max(0.0, float(start)), min(float(data["duration"]), float(end))
    if end - start < MIN_CLIP:
        raise ValueError(f"clip must be at least {MIN_CLIP:g}s long (got {start:.2f}s - {end:.2f}s)")

    out_dir = manifest.parent
    entry = clips[n - 1]
    base = Path(entry["file"]).stem
    words: list[transcribe.Word] = []
    if data.get("words"):
        try:
            words = transcribe.load_words(out_dir / data["words"])
        except (OSError, ValueError, TypeError) as e:
            log(f"can't read the transcript ({e}), re-rendering without captions")
    source = Path(data["source"])
    if not source.is_file():
        # otherwise ffprobe fails and all you get is its command line
        raise ValueError(f"the original video is gone ({source}), was it moved or renamed?")
    info = probe(source)
    log(f"re-rendering {entry['file']}: {start:.1f}s - {end:.1f}s")
    try:
        tmp_name, tmp_sub, length = _render_one(info, words, start, end, out_dir, base,
                                                _render_options(data.get("render")), suffix=".tmp")
    except BaseException:
        for leftover in (f"{base}.tmp.mp4", f"{base}.tmp.ass"):
            (out_dir / leftover).unlink(missing_ok=True)
        raise
    os.replace(out_dir / tmp_name, out_dir / entry["file"])
    if tmp_sub:
        sub = f"{base}.ass"
        os.replace(out_dir / tmp_sub, out_dir / sub)
        entry["subtitles"] = sub
    elif entry.get("subtitles"):
        # the old captions are for the old range, don't leave them looking like they belong to this clip
        (out_dir / Path(entry["subtitles"]).name).unlink(missing_ok=True)
        entry["subtitles"] = None
    entry.update(start=round(start, 2), end=round(end, 2), length=round(length, 2), edited=True)
    _write_manifest(manifest, data)
    return dict(entry)


def mark_clip(manifest: Path, n: int, keep: bool | None) -> dict[str, Any]:
    """Records a keep (True) / drop (False) / undecided (None) choice for clip n in manifest.json."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    clips = data["clips"]
    if not 1 <= n <= len(clips):
        raise ValueError(f"clip {n} doesn't exist, this run has {len(clips)}")
    clips[n - 1]["keep"] = keep
    _write_manifest(manifest, data)
    return dict(clips[n - 1])
