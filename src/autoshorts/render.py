"""Renders one clip to a 1080x1920 MP4 with ffmpeg."""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .probe import MediaInfo, require

Layout = Literal["crop", "blur", "fit"]

W, H = 1080, 1920


@dataclass(frozen=True)
class RenderOptions:
    layout: Layout = "blur"
    encoder: str = "auto"           # auto | h264_nvenc | libx264
    quality: int = 21               # CQ for nvenc, CRF for x264 (lower = better)
    max_fps: int = 60
    normalize_audio: bool = True


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


def video_filter(layout: Layout, subtitle_file: str | None, fps: float, max_fps: int) -> str:
    """filter_complex graph from [0:v] to [vout]."""
    if layout == "crop":
        g = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v]"
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


def build_command(info: MediaInfo, start: float, end: float, out: Path, opts: RenderOptions,
                  subtitle_file: str | None = None) -> list[str]:
    enc = pick_encoder(opts.encoder)
    cmd = [require("ffmpeg"), "-hide_banner", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(info.path),
           "-filter_complex", video_filter(opts.layout, subtitle_file, info.fps, opts.max_fps),
           "-map", "[vout]"]
    # the filter graph reads [0:v], make sure it's the stream we probed (not a cover image)
    cmd[cmd.index("-filter_complex") + 1] = cmd[cmd.index("-filter_complex") + 1].replace(
        "[0:v]", f"[0:{info.video_stream}]", 1)
    if info.has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
        if opts.normalize_audio:
            cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
    if enc == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(opts.quality), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(opts.quality)]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return cmd


def render(info: MediaInfo, start: float, end: float, out: Path, opts: RenderOptions,
           subtitle_file: str | None = None, workdir: Path | None = None) -> None:
    # libass paths on Windows need escaping (C\:/...), running in the subtitle's folder
    # with a bare file name avoids that whole mess
    cmd = build_command(info, start, end, out.resolve(), opts, subtitle_file)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed for {out.name}:\n{tail}")
