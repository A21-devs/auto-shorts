"""Long video in, folder of shorts + manifest.json out."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, scenes, segment, transcribe
from .captions import CaptionStyle, build_ass
from .probe import MediaInfo, probe
from .render import RenderOptions, render

Log = Callable[[str], None]


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


def analyze(info: MediaInfo, job: Job, log: Log) -> tuple[segment.Signals, list[transcribe.Word]]:
    t = time.perf_counter()
    pcm = None
    if info.has_audio:
        log("reading audio")
        pcm = audio.decode_pcm(info.path)
        db = audio.loudness_curve(pcm, hop=job.hop)
        quiet = audio.silences(db, job.hop)
    else:
        db = np.zeros(int(info.duration / job.hop))
        quiet = []
    log("finding scene cuts")
    cuts = scenes.detect_cuts(info.path, info.video_stream)

    words: list[transcribe.Word] = []
    if job.captions and pcm is not None:
        if transcribe.available():
            log(f"transcribing (whisper {job.whisper_model})")
            words = transcribe.transcribe(pcm, job.whisper_model, language=job.language, log=log)
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


def run(job: Job, log: Log = print) -> Path:
    info = probe(job.source)
    log(f"{info.path.name}: {info.duration / 60:.1f} min, {info.width}x{info.height} @ {info.fps:.0f}fps")
    sig, words = analyze(info, job, log)
    clips = segment.find_clips(sig, job.clips)
    if not clips:
        raise RuntimeError("no clips found, is the video shorter than --min?")

    out_dir = job.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    results = []
    for i, clip in enumerate(clips, 1):
        name = f"{stem}_clip{i:02d}.mp4"
        sub = None
        if words:
            sub = f"{stem}_clip{i:02d}.ass"
            (out_dir / sub).write_text(build_ass(words, clip.start, clip.end, CaptionStyle()), encoding="utf-8")
        why = ", ".join(clip.reasons)
        log(f"[{i}/{len(clips)}] {clip.start:7.1f}s - {clip.end:7.1f}s  score {clip.score:5.2f}  {why}")
        t = time.perf_counter()
        render(info, clip.start, clip.end, out_dir / name, job.render, sub, workdir=out_dir)
        log(f"         rendered {name} in {time.perf_counter() - t:.1f}s")
        results.append({**asdict(clip), "file": name, "subtitles": sub})

    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"source": str(info.path), "duration": info.duration, "clips": results}, indent=2),
                        encoding="utf-8")
    return manifest
