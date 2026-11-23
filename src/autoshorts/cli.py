"""autoshorts command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .pipeline import Job, run
from .probe import ToolMissing
from .render import RenderOptions
from .segment import Options


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoshorts", description="Cut a long video into vertical shorts.")
    p.add_argument("video", type=Path, help="input video file")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: <video name>_shorts next to the video)")
    p.add_argument("-n", "--count", type=int, default=6, help="how many clips (default 6)")
    p.add_argument("--min", dest="min_len", type=float, default=20, help="shortest clip in seconds (default 20)")
    p.add_argument("--max", dest="max_len", type=float, default=55, help="longest clip in seconds (default 55)")
    p.add_argument("--layout", choices=["blur", "crop", "follow", "fit"], default="blur",
                   help="blur = video on a blurred background, crop = fill the frame, "
                        "follow = fill the frame and pan toward the action, fit = black bars")
    p.add_argument("--no-captions", action="store_true", help="skip transcription and captions")
    p.add_argument("--model", default="small", help="whisper model size (tiny, base, small, medium, large-v3)")
    p.add_argument("--language", help="force a language code like en (default: auto detect)")
    p.add_argument("--encoder", default="auto", help="auto, h264_nvenc or libx264")
    p.add_argument("--version", action="version", version=f"autoshorts {__version__}")
    return p


def serve_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="autoshorts serve", description="Run the local review server.")
    p.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default 8765, 0 = any free port)")
    args = p.parse_args(argv)
    if not 0 <= args.port <= 65535:
        print("error: --port must be between 0 and 65535", file=sys.stderr)
        return 2
    from .server import serve  # keeps http.server out of the normal CLI path

    try:
        serve(port=args.port)
    except OSError as e:
        print(f"error: can't listen on port {args.port}: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # `serve` is a subcommand, everything else is the original `autoshorts video.mp4 ...` form.
    # A video literally named "serve" can still be passed as ./serve.
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:])
    args = build_parser().parse_args(argv)
    if not args.video.is_file():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    if args.count < 1:
        print("error: --count must be at least 1", file=sys.stderr)
        return 2
    if args.min_len <= 0 or args.max_len < args.min_len:
        print("error: --min must be > 0 and --max must be >= --min", file=sys.stderr)
        return 2
    job = Job(
        source=args.video,
        out_dir=args.out or args.video.with_name(f"{args.video.stem}_shorts"),
        clips=Options(count=args.count, min_len=args.min_len, max_len=args.max_len),
        render=RenderOptions(layout=args.layout, encoder=args.encoder),
        captions=not args.no_captions,
        whisper_model=args.model,
        language=args.language,
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
