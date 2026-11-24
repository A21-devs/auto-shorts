"""Video links in, local files out (yt-dlp, optional).

Anything yt-dlp supports works: YouTube, Twitch VODs, Kick, X, and so on. The file name is
`<title> [<id>].mp4`, so pasting the same link again finds the file that's already there and
skips the download. The modification time is left as the download time (not the upload date),
so the analysis cache still matches on the second run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

Log = Callable[[str], None]
Fraction = Callable[[float], None]

_URL = re.compile(r"https?://\S+", re.IGNORECASE)

# h264 + aac in mp4 when the site has it (cuts and decodes fastest), never above 1080p:
# the shorts are 1080 wide anyway, and a 4K VOD is 4x the download for nothing.
FORMAT = ("bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b")


class DownloadError(RuntimeError):
    pass


def is_url(text: str) -> bool:
    return bool(_URL.fullmatch(text.strip()))


def available() -> bool:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return True


def options(folder: Path, log: Log, on_progress: Fraction | None, subtitles: str | None) -> dict[str, Any]:
    def hook(d: dict[str, Any]) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    class Logger:
        def debug(self, msg: str) -> None:
            if msg.startswith("[download] ") and "has already been downloaded" in msg:
                log("already downloaded, reusing the file")

        def info(self, msg: str) -> None:
            pass

        def warning(self, msg: str) -> None:
            log(f"yt-dlp: {msg}")

        def error(self, msg: str) -> None:
            log(f"yt-dlp: {msg}")

    opts: dict[str, Any] = {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": "%(title).80B [%(id)s].%(ext)s",
        "paths": {"home": str(folder)},
        "windowsfilenames": True,
        "updatetime": False,
        "noplaylist": True,
        "progress_hooks": [hook],
        "logger": Logger(),
    }
    if subtitles:
        # creator subtitles if there are any, auto captions otherwise. clipwright reads
        # <video>.srt next to the video, so it's renamed to that after the download.
        opts.update(writesubtitles=True, writeautomaticsub=True, subtitleslangs=[subtitles],
                    subtitlesformat="srt/best",
                    postprocessors=[{"key": "FFmpegSubtitlesConvertor", "format": "srt"}])
    return opts


def download(url: str, folder: Path, log: Log = print, on_progress: Fraction | None = None,
             subtitles: str | None = None) -> Path:
    """Downloads url into folder and returns the video file. subtitles: a language code like
    "en" to also fetch subtitles as <video>.srt, or None to skip them."""
    if not available():
        raise DownloadError('links need yt-dlp: pip install "clipwright[link]"')
    import yt_dlp

    folder.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    with yt_dlp.YoutubeDL(options(folder, log, on_progress, subtitles)) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise DownloadError(str(e).removeprefix("ERROR: ")) from None
        if info is None:
            raise DownloadError(f"nothing to download at {url}")
        video = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not video.is_file():
        raise DownloadError(f"yt-dlp finished but {video.name} isn't there")
    if subtitles:
        _rename_subtitles(video, subtitles)
    return video


def _rename_subtitles(video: Path, lang: str) -> None:
    """`<video>.en.srt` (or `.en-orig.srt`, `.en-US.srt`, ...) -> `<video>.srt`."""
    target = video.with_suffix(".srt")
    stem = video.with_suffix("").name
    for sub in sorted(video.parent.glob(f"{glob_escape(stem)}.{lang}*.srt")):
        sub.replace(target)
        break


def glob_escape(name: str) -> str:
    # video titles are full of [ ] which glob treats as character classes
    return re.sub(r"([\[\]*?])", r"[\1]", name)
