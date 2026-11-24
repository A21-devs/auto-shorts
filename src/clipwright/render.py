"""Renders one clip to a 1080x1920 MP4 with ffmpeg."""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import follow, jumpcut, motion
from .probe import MediaInfo, require

Layout = Literal["crop", "blur", "fit", "follow"]

W, H = 1080, 1920
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class RenderOptions:
    layout: Layout = "blur"
    encoder: str = "auto"           # auto | h264_nvenc | libx264
    quality: int = 21               # CQ for nvenc, CRF for x264 (lower = better)
    max_fps: int = 60
    normalize_audio: bool = True
    jumpcut: bool = False           # cut out pauses inside the clip (needs audio)


@functools.cache
def has_encoder(name: str) -> bool:
    out = subprocess.run([require("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True)
    return f" {name} " in out.stdout


@functools.cache
def encoder_works(name: str) -> bool:
    """Being listed by `ffmpeg -encoders` doesn't mean it runs: NVENC is compiled into most Windows
    builds but fails on machines without an NVIDIA GPU or with an older driver. Encode a few
    frames to find out for real."""
    if not has_encoder(name):
        return False
    proc = subprocess.run(
        [require("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
         "-c:v", name, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def pick_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    return "h264_nvenc" if encoder_works("h264_nvenc") else "libx264"


def video_filter(layout: Layout, subtitle_file: str | None, fps: float, max_fps: int,
                 crop_x: str | None = None) -> str:
    """filter_complex graph from [0:v] to [vout]. `crop_x` is the x expression for "follow"
    (without one it behaves like "crop")."""
    if layout == "crop" or (layout == "follow" and not crop_x):
        g = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v]"
    elif layout == "follow":
        g = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
             f"crop=w={W}:h={H}:x='{crop_x}':y='(ih-oh)/2',setsar=1[v]")
    elif layout == "fit":
        g = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v]")
    else:  # blur: the video fits the width, a blurred zoomed copy fills the rest
        g = (f"[0:v]split[a][b];"
             f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
             f"boxblur=10:2,eq=brightness=-0.12,scale={W}:{H}[bg];"
             f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
             f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]")
    last = "v"
    if fps > max_fps:
        g += f";[{last}]fps={max_fps}[vf]"
        last = "vf"
    if subtitle_file:
        g += f";[{last}]ass={subtitle_file}[vout]"
    else:
        g += f";[{last}]null[vout]"
    return g


TARGET_LUFS = -14.0
PEAK_LIMIT = 0.841  # -1.5 dBFS, linear
MAX_GAIN_DB = 24.0


def parse_integrated(stderr: str) -> float | None:
    """input_i from the JSON block loudnorm prints in measuring mode. None for silence."""
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(stderr[start:end + 1]).get("input_i")
        i = float(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return i if i > -70 else None  # -inf / near-silent: nothing to normalize


def gain_chain(gain_db: float) -> str:
    """Make-up gain followed by a peak limiter.

    Plain loudnorm can't hit -14 LUFS on peaky speech: it refuses to push true peak past the
    limit and silently drops to its cautious dynamic mode (measured -18 to -19 LUFS on test
    clips). Gain + limiter gets to the target and keeps peaks at -1.5 dBFS.
    """
    return f"volume={gain_db:.2f}dB,alimiter=limit={PEAK_LIMIT}:attack=5:release=50:level=disabled"


def measure(info: MediaInfo, start: float, end: float, chain: str | None = None) -> float | None:
    af = f"{chain + ',' if chain else ''}loudnorm=I={TARGET_LUFS}:print_format=json"
    proc = subprocess.run(
        [require("ffmpeg"), "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
         "-i", str(info.path), "-map", "0:a:0", "-af", af, "-f", "null", "-"],
        capture_output=True, text=True, errors="replace",
    )
    return parse_integrated(proc.stderr) if proc.returncode == 0 else None


def fit_gain(info: MediaInfo, start: float, end: float) -> float | None:
    """Gain that lands the clip on TARGET_LUFS after limiting. The limiter eats some loudness
    on peaky audio, so measure once more through the chain and correct. Audio-only passes run
    at ~40x realtime, so this is well under a second per clip."""
    loud = measure(info, start, end)
    if loud is None:
        return None
    gain = TARGET_LUFS - loud
    # the more gain, the more the limiter eats, so a single correction undershoots (-15.5 on
    # test clips). A few cheap iterations converge to within half a dB.
    for _ in range(4):
        after = measure(info, start, end, gain_chain(gain))
        if after is None or abs(TARGET_LUFS - after) < 0.5:
            break
        gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain + (TARGET_LUFS - after)))
    return gain


def jumpcut_filter(keep: jumpcut.Ranges, audio: bool, gain: str | None) -> str:
    """Graph that keeps only the `keep` ranges (clip-relative) of [0:v] (and [0:a:0]) and joins
    them, ending in [jv] (and [aout], with the gain chain applied).

    trim + concat rather than select/setpts: select needs setpts=N/FRAME_RATE/TB to close the
    gaps, which drifts against the audio on variable frame rate recordings."""
    n = len(keep)
    g = f"[0:v]split={n}" + "".join(f"[s{i}]" for i in range(n)) + ";"
    if audio:
        g += f"[0:a:0]asplit={n}" + "".join(f"[t{i}]" for i in range(n)) + ";"
    pairs = ""
    for i, (a, b) in enumerate(keep):
        g += f"[s{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}];"
        if audio:
            g += f"[t{i}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}];"
        pairs += f"[v{i}][a{i}]" if audio else f"[v{i}]"
    if audio:
        g += f"{pairs}concat=n={n}:v=1:a=1[jv][ja];[ja]{gain or 'anull'}[aout]"
    else:
        g += f"{pairs}concat=n={n}:v=1:a=0[jv]"
    return g


def build_command(info: MediaInfo, start: float, end: float, out: Path, opts: RenderOptions,
                  subtitle_file: str | None = None, gain_db: float | None = None,
                  crop_x: str | None = None, keep: jumpcut.Ranges | None = None) -> list[str]:
    """`keep`: clip-relative ranges to keep (jump cuts). None, or one range covering the whole
    clip, renders it straight through."""
    enc = pick_encoder(opts.encoder)
    gain = gain_chain(gain_db) if info.has_audio and opts.normalize_audio and gain_db is not None else None
    graph = video_filter(opts.layout, subtitle_file, info.fps, opts.max_fps, crop_x)
    cut = keep is not None and len(keep) > 0 and keep != [(0.0, round(end - start, 3))]
    if keep is not None and cut:
        graph = jumpcut_filter(keep, info.has_audio, gain) + ";" + graph.replace("[0:v]", "[jv]", 1)
    # the filter graph reads [0:v], make sure it's the stream we probed (not a cover image)
    graph = graph.replace("[0:v]", f"[0:{info.video_stream}]", 1)
    cmd = [require("ffmpeg"), "-hide_banner", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(info.path), "-filter_complex", graph, "-map", "[vout]"]
    if info.has_audio:
        cmd += ["-map", "[aout]" if cut else "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
        if gain and not cut:
            cmd += ["-af", gain]
    if enc == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(opts.quality), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(opts.quality)]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return cmd


def follow_crop_x(info: MediaInfo, start: float, end: float, keep: jumpcut.Ranges | None = None) -> str | None:
    """crop x expression that pans toward the action in [start, end], or None when the source is
    already 9:16 or narrower (nothing to pan). With jump cuts the keyframes are moved onto the
    shortened timeline, since crop sees the frames after concat."""
    scaled_w = info.width * H / info.height
    if scaled_w <= W + 1:
        return None
    m = motion.read_motion(info.path, info.video_stream, info.width, info.height, start, end - start)
    keys = follow.camera_path(motion.focus(m, window=1.0), step=1.0, window=W / scaled_w)
    if keep:
        keys = [(jumpcut.retime(t, keep), c) for t, c in keys]
    return follow.crop_x_expr(keys)


def render(info: MediaInfo, start: float, end: float, out: Path, opts: RenderOptions,
           subtitle_file: str | None = None, workdir: Path | None = None,
           keep: jumpcut.Ranges | None = None) -> None:
    # libass paths on Windows need escaping (C\:/...), running in the subtitle's folder
    # with a bare file name avoids that. The name itself also goes into the filter graph,
    # where , ; ' [ ] and : mean something, so anything unusual gets a safe temp name.
    temp_sub: Path | None = None
    if subtitle_file and not SAFE_NAME.fullmatch(subtitle_file):
        base = workdir or Path.cwd()
        temp_sub = base / f"_clipwright_{out.stem.encode().hex()[:16]}.ass"
        shutil.copyfile(base / subtitle_file, temp_sub)
        subtitle_file = temp_sub.name
    gain = fit_gain(info, start, end) if info.has_audio and opts.normalize_audio else None
    crop_x = follow_crop_x(info, start, end, keep) if opts.layout == "follow" else None
    cmd = build_command(info, start, end, out.resolve(), opts, subtitle_file, gain, crop_x, keep)
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, errors="replace")
    finally:
        if temp_sub:
            temp_sub.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed for {out.name}:\n{tail}")
