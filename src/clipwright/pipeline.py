"""Long video in, folder of shorts + manifest.json out."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from . import aipick, audio, copywriting, jumpcut, platforms, scenes, segment, speaker, subtitles, thumbnail, transcribe
from .captions import STYLES, build_ass, build_srt, hook_text
from .probe import MediaInfo, probe
from .render import LAYOUTS, Layout, RenderOptions, frame_size, render

Log = Callable[[str], None]
# (stage, fraction of that stage done). Stages in order: audio, scenes, transcribe (only with
# captions), render, done. Fractions go from 0 to 1 and never go down within a stage.
Progress = Callable[[str, float], None]

MIN_CLIP = 1.0  # shortest range rerender_clip accepts, in seconds


@dataclass
class Job:
    source: Path
    # None: <video name>_shorts next to the video
    out_dir: Path | None = None
    clips: segment.Options = field(default_factory=segment.Options)
    render: RenderOptions = field(default_factory=RenderOptions)
    captions: bool = True
    whisper_model: str = "small"
    language: str | None = None
    hop: float = 0.5
    # existing subtitles to use instead of Whisper (also works when the audio track is silent)
    srt: Path | None = None
    # what the clips are cut for (platforms.PLATFORMS key), recorded in the manifest
    platform: str | None = None
    # the link the source was downloaded from, kept in the manifest
    url: str | None = None
    # write a title, description and hashtags per clip with an AI: a copywriting.BACKENDS name
    copy: str | None = None
    # model name for the ollama and openai backends (None: ollama's default, openai's from the environment)
    copy_model: str | None = None
    # let an AI rate a shortlist of moments by their transcript (aipick.py): a copywriting.BACKENDS name
    pick: str | None = None


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
    if job.captions and job.srt is not None:
        words = subtitles.load_srt(job.srt)
        log(f"using subtitles from {job.srt.name} ({len(words)} words), skipping whisper")
        if not words:
            log("warning: no subtitle cues found in that file, so no captions")
        elif words[-1].end > info.duration + 5:
            log(f"warning: the subtitles run to {words[-1].end:.0f}s but the video is {info.duration:.0f}s long, "
                "they may be for a different version of it")
    elif job.captions and pcm is not None:
        if transcribe.available():
            log(f"transcribing (whisper {job.whisper_model})")
            progress("transcribe", 0.0)
            words = transcribe.transcribe(pcm, job.whisper_model, language=job.language, log=log,
                                          on_progress=lambda f: progress("transcribe", f))
            progress("transcribe", 1.0)
        else:
            log("faster-whisper not installed, skipping captions (pip install clipwright[transcribe])")

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
    """Deletes <stem>_clipNN.mp4/.ass/.srt/.jpg/.txt with NN > keep. Only touches files that match our own naming."""
    pattern = re.compile(re.escape(stem) + r"_clip(\d{2,})\.(mp4|ass|srt|jpg|txt)")
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


class Rendered(NamedTuple):
    file: str
    subtitles: str | None   # .ass burned into the video, None without words
    srt: str | None         # upload subtitles, None when no words fall in the clip
    thumbnail: str | None   # cover .jpg, None if it couldn't be made
    length: float           # seconds, shorter than end - start with jump cuts


def _render_one(info: MediaInfo, words: list[transcribe.Word], start: float, end: float, out_dir: Path,
                base: str, opts: RenderOptions, suffix: str = "", log: Log = print,
                hook: str | None = None) -> Rendered:
    """Writes <base><suffix>.ass (if there are words or a hook title) and .srt (if any words are in
    this clip), renders <base><suffix>.mp4 and renders its cover <base><suffix>.jpg."""
    name = f"{base}{suffix}.mp4"
    keep = plan_cuts(info, start, end) if opts.jumpcut else None
    length = jumpcut.kept_duration(keep) if keep else end - start
    sub = srt = None
    if words or hook:
        sub = f"{base}{suffix}.ass"
        if keep:
            timed, t0, t1 = jumpcut.retime_words(words, start, keep), 0.0, length
        else:
            timed, t0, t1 = words, start, end
        w, h = frame_size(opts.aspect)
        style = STYLES.get(opts.caption_style, STYLES["bold"]).for_frame(w, h)
        ass = build_ass(timed, t0, t1, style, width=w, height=h, hook=hook)
        (out_dir / sub).write_text(ass, encoding="utf-8")
        srt_text = build_srt(timed, t0, t1)
        if srt_text:  # a 0-byte .srt gets rejected on upload, no file is better
            srt = f"{base}{suffix}.srt"
            (out_dir / srt).write_text(srt_text, encoding="utf-8")
    crop_x = render(info, start, end, out_dir / name, opts, sub, workdir=out_dir, keep=keep)
    cover = f"{base}{suffix}.jpg"
    thumb: str | None = cover
    try:
        thumbnail.make_cover(out_dir / name, info, start, out_dir / cover, opts, keep, crop_x)
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as e:
        # a missing cover is no reason to throw away a finished clip. CalledProcessError's text is
        # the whole ffprobe command line, which says nothing a person can use
        why = "ffprobe couldn't read it" if isinstance(e, subprocess.CalledProcessError) else str(e).splitlines()[0]
        log(f"         no thumbnail for {name} ({why})")
        thumb = None
    return Rendered(name, sub, srt, thumb, length)


def _post_copy(job: Job, words: list[transcribe.Word], start: float, end: float,
               log: Log) -> tuple[dict[str, Any] | None, bool]:
    """The post text for one clip, and whether it's worth trying the next clip."""
    inside = [w for w in words if w.end > start and w.start < end]
    assert job.copy is not None
    try:
        post = copywriting.write_copy(inside, job.copy, job.platform, job.copy_model)
    except copywriting.CopyError as e:
        log(f"         no title/hashtags: {e}")
        return None, not e.setup
    log(f"         title: {post.title}")
    return post.to_json(), True


def post_text(post: dict[str, Any]) -> str:
    """The text to paste when posting: title, description, then the hashtags on one line. Platforms
    without a title field (TikTok, Reels, X) take the whole thing as the caption."""
    tags = " ".join(f"#{t}" for t in post.get("hashtags") or [])
    parts = (str(post.get("title") or ""), str(post.get("description") or ""), tags)
    return "\n\n".join(p for p in parts if p) + "\n"


def _write_post(path: Path, post: dict[str, Any]) -> str:
    path.write_text(post_text(post), encoding="utf-8")
    return path.name


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    # write then rename, so a reader (the review page) never sees half a file
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run(job: Job, log: Log = print, progress: Progress = _no_progress) -> Path:
    info = probe(job.source)
    log(f"{info.path.name}: {info.duration / 60:.1f} min, {info.width}x{info.height} @ {info.fps:.0f}fps")
    sig, words = analyze(info, job, log, progress)
    if job.pick:
        backend, model = job.pick, job.copy_model
        clips = aipick.find_clips(
            sig, job.clips, words,
            lambda prompt, system, schema: copywriting.ask_json(backend, prompt, system, schema, model),
            job.platform, log)
    else:
        clips = segment.find_clips(sig, job.clips)
    if not clips:
        raise RuntimeError("no clips found, is the video shorter than --min?")

    out_dir = job.out_dir or info.path.with_name(f"{info.path.stem}_shorts")
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
    copy = job.copy  # set to None after a failure that would repeat on every clip
    for i, clip in enumerate(clips, 1):
        why = ", ".join(clip.reasons)
        log(f"[{i}/{len(clips)}] {clip.start:7.1f}s - {clip.end:7.1f}s  score {clip.score:5.2f}  {why}")
        t = time.perf_counter()
        post: dict[str, Any] | None = None
        if copy:  # before the render, so a --hook can use the AI title
            post, ok = _post_copy(job, words, clip.start, clip.end, log)
            if not ok:
                copy = None
                log("         skipping titles for the rest of the clips")
        hook = None
        if job.render.hook:
            hook = str(post["title"]) if post else hook_text(words, clip.start, clip.end)
        r = _render_one(info, words, clip.start, clip.end, out_dir, f"{stem}_clip{i:02d}", job.render, log=log,
                        hook=hook)
        name = r.file
        # round once, so the log and the review page (which reads the manifest) show the same number
        length = round(r.length, 2)
        cut = clip.duration - length
        log(f"         rendered {name} in {time.perf_counter() - t:.1f}s"
            + (f", cut {cut:.1f}s of pauses ({length:.1f}s left)" if cut > 0.05 else ""))
        if cut > 0.05 and length < job.clips.min_len - 0.05:
            # clips are picked by range length, the cuts happen after that. Without cuts a clip
            # can only be under --min when the whole video is, which the pick already says.
            log(f"         warning: {name} is {length:.1f}s after cutting pauses, under --min {job.clips.min_len:g}s")
        entry = {**asdict(clip), "file": name, "subtitles": r.subtitles, "srt": r.srt,
                 "thumbnail": r.thumbnail, "length": length}
        if job.copy:
            entry["copy"] = post
            entry["post"] = None
            if post:
                entry["post"] = _write_post(out_dir / f"{stem}_clip{i:02d}.txt", post)
        if hook:
            entry["hook"] = hook
        results.append(entry)
        progress("render", i / len(clips))

    manifest = out_dir / "manifest.json"
    _write_manifest(manifest, {"source": str(info.path), "duration": info.duration, "platform": job.platform,
                               "url": job.url, "pick": job.pick,
                               "render": asdict(job.render),
                               "words": words_file, "clips": results})
    progress("done", 1.0)
    return manifest


def _render_options(saved: dict[str, Any] | None) -> RenderOptions:
    """RenderOptions from a manifest. Older manifests don't have them, unknown keys are ignored."""
    known = {f.name for f in fields(RenderOptions)}
    return RenderOptions(**{k: v for k, v in (saved or {}).items() if k in known})


def check_layout(layout: object) -> Layout:
    """`layout` as a Layout, or ValueError with what's wrong (unknown, or speaker without OpenCV)."""
    for known in LAYOUTS:
        if layout == known:
            if known == "speaker" and not speaker.available():
                raise ValueError('the speaker layout needs OpenCV: pip install "clipwright[speaker]"')
            return known
    raise ValueError(f"layout must be one of {', '.join(LAYOUTS)}")


def rerender_clip(manifest: Path, n: int, start: float, end: float, log: Log = print,
                  layout: str | None = None) -> dict[str, Any]:
    """Re-renders clip number n (1-based) of a finished run over [start, end] and updates manifest.json.

    `layout` changes the layout of this clip only. It's saved on the clip's entry, so later
    re-renders of it (a trim on the review page) keep it; other clips keep the run's layout.

    Renders to a temp name first and swaps it in at the end, so a failed render leaves the old clip
    (and its captions) as they were.
    """
    data = json.loads(manifest.read_text(encoding="utf-8"))
    clips = data["clips"]
    if not 1 <= n <= len(clips):
        raise ValueError(f"clip {n} doesn't exist, this run has {len(clips)}")
    if layout is not None:
        layout = check_layout(layout)
    start, end = max(0.0, float(start)), min(float(data["duration"]), float(end))
    if end - start < MIN_CLIP:
        raise ValueError(f"clip must be at least {MIN_CLIP:g}s long (got {start:.2f}s - {end:.2f}s)")
    target = platforms.PLATFORMS.get(data.get("platform") or "")
    if target and end - start > target.limit:
        raise ValueError(f"{target.name} clips can be at most {target.limit:g}s (this range is {end - start:.1f}s)")

    out_dir = manifest.parent
    entry = clips[n - 1]
    base = Path(entry["file"]).stem
    opts = _render_options(data.get("render"))
    chosen = layout or entry.get("layout")
    if chosen:
        opts = replace(opts, layout=check_layout(chosen))
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
    log(f"re-rendering {entry['file']}: {start:.1f}s - {end:.1f}s, {opts.layout} layout")
    try:
        r = _render_one(info, words, start, end, out_dir, base, opts, suffix=".tmp", log=log,
                        hook=entry.get("hook") if opts.hook else None)
    except BaseException:
        for leftover in (f"{base}.tmp.mp4", f"{base}.tmp.ass", f"{base}.tmp.srt", f"{base}.tmp.jpg"):
            (out_dir / leftover).unlink(missing_ok=True)
        raise
    os.replace(out_dir / r.file, out_dir / entry["file"])
    # the old captions and cover are for the old range, don't leave them looking like they belong to this clip
    swaps = (("subtitles", r.subtitles, ".ass"), ("srt", r.srt, ".srt"), ("thumbnail", r.thumbnail, ".jpg"))
    for key, tmp, ext in swaps:
        if tmp:
            os.replace(out_dir / tmp, out_dir / f"{base}{ext}")
            entry[key] = f"{base}{ext}"
        else:
            if entry.get(key):
                (out_dir / Path(entry[key]).name).unlink(missing_ok=True)
            entry[key] = None
    entry.update(start=round(start, 2), end=round(end, 2), length=round(r.length, 2), edited=True)
    if layout is not None:
        entry["layout"] = layout
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
