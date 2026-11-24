"""clipwright command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, download
from .pipeline import Job, run
from .platforms import PLATFORMS, lengths
from .probe import ToolMissing
from .render import RenderOptions
from .segment import Options
from .subtitles import sidecar


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipwright", description="Cut a long video into vertical shorts.")
    p.add_argument("video", help="input video file, or a link (YouTube, Twitch, ...) with clipwright[link] installed")
    p.add_argument("--downloads", type=Path, default=Path("."),
                   help="where links are downloaded to (default: the current folder)")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: <video name>_shorts next to the video)")
    p.add_argument("-n", "--count", type=int, default=6, help="how many clips (default 6)")
    p.add_argument("--platform", choices=list(PLATFORMS),
                   help="pick clip lengths for this platform: "
                        + ", ".join(f"{k} {v.min_len:g}-{v.max_len:g}s" for k, v in PLATFORMS.items()))
    p.add_argument("--min", dest="min_len", type=float, help="shortest clip in seconds (default 20, or the platform's)")
    p.add_argument("--max", dest="max_len", type=float, help="longest clip in seconds (default 55, or the platform's)")
    p.add_argument("--layout", choices=["blur", "crop", "follow", "fit"], default="blur",
                   help="blur = video on a blurred background, crop = fill the frame, "
                        "follow = fill the frame and pan toward the action, fit = black bars")
    p.add_argument("--no-captions", action="store_true", help="skip transcription and captions")
    p.add_argument("--jumpcut", action="store_true",
                   help="cut out pauses longer than 0.5s inside each clip (captions are retimed to match)")
    p.add_argument("--srt", type=Path,
                   help="use this subtitle file instead of transcribing (default: <video name>.srt next to the "
                        "video, if there is one)")
    p.add_argument("--model", default="small", help="whisper model size (tiny, base, small, medium, large-v3)")
    p.add_argument("--language", help="force a language code like en (default: auto detect)")
    p.add_argument("--encoder", default="auto", help="auto, h264_nvenc or libx264")
    p.add_argument("--version", action="version", version=f"clipwright {__version__}")
    return p


def serve_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clipwright serve", description="Run the local review server.")
    p.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765, 0 = any free port)")
    p.add_argument("--downloads", type=Path, help="where pasted links are downloaded to (default: Videos/clipwright)")
    args = p.parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("error: --port must be between 0 and 65535", file=sys.stderr)
        return 2
    from .server import serve  # keeps http.server out of the normal CLI path

    try:
        serve(port=args.port, downloads=args.downloads)
    except OSError as e:
        print(f"error: can't listen on port {args.port}: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # `serve` is a subcommand, everything else is the original `clipwright video.mp4 ...` form.
    # A video literally named "serve" can still be passed as ./serve.
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:])
    args = build_parser().parse_args(argv)
    url = args.video if download.is_url(args.video) else None
    args.video = Path(args.video)
    if url is None and not args.video.is_file():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    if url is not None and not download.available():
        print('error: links need yt-dlp: pip install "clipwright[link]"', file=sys.stderr)
        return 2
    if args.count < 1:
        print("error: --count must be at least 1", file=sys.stderr)
        return 2
    try:
        args.min_len, args.max_len = lengths(args.platform, args.min_len, args.max_len)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.min_len <= 0 or args.max_len < args.min_len:
        print("error: --min must be > 0 and --max must be >= --min", file=sys.stderr)
        return 2
    srt = args.srt
    if srt is not None and not srt.is_file():
        print(f"error: {srt} not found", file=sys.stderr)
        return 2
    if url is not None:
        # checked everything else first, so a typo in --max doesn't cost a 2 GB download
        try:
            args.video = download.download(url, args.downloads,
                                           subtitles=None if args.no_captions or srt else (args.language or "en"))
        except download.DownloadError as e:
            print(f"error: {e}", file=sys.stderr)
            return 4
    if srt is None and not args.no_captions:
        srt = sidecar(args.video)
    job = Job(
        source=args.video,
        out_dir=args.out,
        clips=Options(count=args.count, min_len=args.min_len, max_len=args.max_len),
        render=RenderOptions(layout=args.layout, encoder=args.encoder, jumpcut=args.jumpcut),
        captions=not args.no_captions,
        whisper_model=args.model,
        language=args.language,
        srt=srt,
        platform=args.platform,
        url=url,
    )
    try:
        manifest = run(job)
    except ToolMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(f"\ndone: {manifest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
