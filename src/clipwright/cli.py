"""clipwright command line."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from . import __version__, copywriting, download, speaker
from .captions import STYLES
from .pipeline import Job, run
from .platforms import PLATFORMS, aspect_for, lengths
from .probe import ToolMissing, require
from .render import ASPECTS, LAYOUTS, RenderOptions
from .segment import Options
from .subtitles import sidecar


def add_job_args(p: argparse.ArgumentParser) -> None:
    """Options that shape each job, shared by the normal run and `watch`."""
    p.add_argument("-n", "--count", type=int, default=6, help="how many clips (default 6)")
    p.add_argument("--platform", choices=list(PLATFORMS),
                   help="pick clip lengths for this platform: "
                        + ", ".join(f"{k} {v.min_len:g}-{v.max_len:g}s" for k, v in PLATFORMS.items()))
    p.add_argument("--min", dest="min_len", type=float, help="shortest clip in seconds (default 20, or the platform's)")
    p.add_argument("--max", dest="max_len", type=float, help="longest clip in seconds (default 55, or the platform's)")
    p.add_argument("--layout", choices=LAYOUTS, default="blur",
                   help="blur = video on a blurred background, crop = fill the frame, "
                        "follow = fill the frame and pan toward the action, speaker = fill the frame and cut to "
                        "whoever is talking (podcasts, interviews; needs clipwright[speaker]), fit = black bars")
    p.add_argument("--aspect", choices=list(ASPECTS),
                   help="output frame: 9:16 vertical (default), 1:1 square, 4:5 feed, 16:9 landscape. "
                        "--platform youtube/x pick 16:9 and linkedin 1:1 unless this is given")
    p.add_argument("--no-captions", action="store_true", help="skip transcription and captions")
    p.add_argument("--caption-style", choices=list(STYLES), default="bold",
                   help="bold (default, all caps with a yellow word), clean (mixed case, green), boxed (on a dark "
                        "box, for busy gameplay), pop (huge, two words at a time), minimal (small, no highlight)")
    p.add_argument("--hook", action="store_true",
                   help="show a title at the top for the first 3 seconds: the --copy title if there is one, "
                        "otherwise the clip's first sentence")
    p.add_argument("--jumpcut", action="store_true",
                   help="cut out pauses longer than 0.5s inside each clip (captions are retimed to match)")
    p.add_argument("--pick", choices=copywriting.BACKENDS,
                   help="let an AI read the transcript of the best candidate moments and help choose the clips: "
                        "claude (Claude Code CLI, no API key), ollama, or openai (LM Studio or any OpenAI-compatible "
                        "server, see CLIPWRIGHT_OPENAI_URL). Only transcript text is sent")
    p.add_argument("--copy", choices=copywriting.BACKENDS,
                   help="write a title, description and hashtags for each clip with an AI (same choices as "
                        "--pick). Only the clip's transcript is sent")
    p.add_argument("--copy-model", default=None,
                   help=f"model for ollama or openai with --copy/--pick (ollama default "
                        f"{copywriting.DEFAULT_OLLAMA_MODEL}, openai default $CLIPWRIGHT_OPENAI_MODEL)")
    p.add_argument("--model", default="small", help="whisper model size (tiny, base, small, medium, large-v3)")
    p.add_argument("--language", help="force a language code like en (default: auto detect)")
    p.add_argument("--encoder", default="auto", help="auto, h264_nvenc or libx264")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipwright", description="Cut a long video into short clips.")
    p.add_argument("video", help="input video file, or a link (YouTube, Twitch, ...) with clipwright[link] installed")
    p.add_argument("--downloads", type=Path, default=Path("."),
                   help="where links are downloaded to (default: the current folder)")
    p.add_argument("-o", "--out", type=Path, help="output folder (default: <video name>_shorts next to the video)")
    add_job_args(p)
    p.add_argument("--srt", type=Path,
                   help="use this subtitle file instead of transcribing (default: <video name>.srt next to the "
                        "video, if there is one)")
    p.add_argument("--version", action="version", version=f"clipwright {__version__}")
    return p


def check_job_args(args: argparse.Namespace) -> str | None:
    """Validates the shared job options (filling in min/max from --platform). Returns an error
    message, or None when they're fine."""
    if args.layout == "speaker" and not speaker.available():
        return '--layout speaker needs OpenCV: pip install "clipwright[speaker]"'
    if args.copy and args.no_captions:
        return "--copy writes from the transcript, so it can't be used with --no-captions"
    if args.pick and args.no_captions:
        return "--pick reads the transcript, so it can't be used with --no-captions"
    if args.count < 1:
        return "--count must be at least 1"
    try:
        args.min_len, args.max_len = lengths(args.platform, args.min_len, args.max_len)
    except ValueError as e:
        return str(e)
    if args.min_len <= 0 or args.max_len < args.min_len:
        return "--min must be > 0 and --max must be >= --min"
    args.aspect = aspect_for(args.platform, args.aspect)
    return None


def job_from_args(args: argparse.Namespace, video: Path, out_dir: Path | None, srt: Path | None,
                  url: str | None = None) -> Job:
    return Job(
        source=video,
        out_dir=out_dir,
        clips=Options(count=args.count, min_len=args.min_len, max_len=args.max_len),
        render=RenderOptions(layout=args.layout, encoder=args.encoder, jumpcut=args.jumpcut, aspect=args.aspect,
                             caption_style=args.caption_style, hook=args.hook),
        captions=not args.no_captions,
        whisper_model=args.model,
        language=args.language,
        srt=srt,
        platform=args.platform,
        url=url,
        copy=args.copy,
        copy_model=args.copy_model,
        pick=args.pick,
    )


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


def watch_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="clipwright watch",
        description="Wait for new recordings in a folder (OBS and the like) and clip each one when it's finished. "
                    "Recordings that already have clips are skipped, so it's safe to restart.")
    p.add_argument("folder", type=Path, help="folder the recorder saves to (subfolders are not watched)")
    p.add_argument("-o", "--out", type=Path,
                   help="put each recording's <name>_shorts folder here (default: next to the recording)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="seconds a file has to stop changing before it counts as finished (default 10)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between folder checks (default 2)")
    add_job_args(p)
    args = p.parse_args(argv)
    if not args.folder.is_dir():
        print(f"error: {args.folder} is not a folder", file=sys.stderr)
        return 2
    if args.settle < 0 or args.interval <= 0:
        print("error: --settle must be >= 0 and --interval > 0", file=sys.stderr)
        return 2
    if (err := check_job_args(args)) is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    try:
        # the normal run finds out on the first ffprobe call, a watcher would only log a failure
        # per recording and look healthy
        require("ffmpeg")
        require("ffprobe")
    except ToolMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    if isinstance(sys.stdout, io.TextIOWrapper):
        # a watcher usually runs with its output going to a file (Task Scheduler, `> watch.log`),
        # where Python would buffer it in 8 KB blocks and lose the tail if it's stopped
        sys.stdout.reconfigure(line_buffering=True)
    from . import watch  # only needed here

    def handle(video: Path) -> None:
        srt = None if args.no_captions else sidecar(video)
        manifest = run(job_from_args(args, video, watch.out_dir_for(video, args.out), srt))
        print(f"done: {manifest.parent}\n")

    print(f"watching {args.folder.resolve()} (Ctrl+C to stop)")
    try:
        watch.watch(args.folder, handle, settle=args.settle, interval=args.interval, out_root=args.out)
    except KeyboardInterrupt:
        print("stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # `serve`, `watch` and `mcp` are subcommands, everything else is the original
    # `clipwright video.mp4 ...` form. A video literally named "serve" can still be passed as ./serve.
    if argv[:1] == ["serve"]:
        return serve_main(argv[1:])
    if argv[:1] == ["watch"]:
        return watch_main(argv[1:])
    if argv[:1] == ["mcp"]:
        from .mcp_server import main as mcp_main  # keeps the MCP SDK out of the normal CLI path

        return mcp_main()
    args = build_parser().parse_args(argv)
    url = args.video if download.is_url(args.video) else None
    args.video = Path(args.video)
    if url is None and not args.video.is_file():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    if url is not None and not download.available():
        print('error: links need yt-dlp: pip install "clipwright[link]"', file=sys.stderr)
        return 2
    if (err := check_job_args(args)) is not None:
        print(f"error: {err}", file=sys.stderr)
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
    job = job_from_args(args, args.video, args.out, srt, url)
    try:
        manifest = run(job)
    except ToolMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(f"\ndone: {manifest.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
