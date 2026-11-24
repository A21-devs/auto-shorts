import sys
import types
from pathlib import Path
from typing import Any

import pytest

from clipwright import download
from clipwright.cli import main


@pytest.mark.parametrize(("text", "ok"), [
    ("https://www.youtube.com/watch?v=abc", True),
    ("  http://twitch.tv/videos/123  ", True),
    ("HTTPS://X.COM/a/status/1", True),
    ("C:\\Videos\\stream.mp4", False),
    ("https://a.com/x y", False),
    ("youtube.com/watch?v=abc", False),
])
def test_is_url(text: str, ok: bool):
    assert download.is_url(text) is ok


def test_subtitles_renamed_to_the_sidecar_name(tmp_path: Path):
    video = tmp_path / "Talk [part 1] [abc].mp4"
    video.write_bytes(b"v")
    (tmp_path / "Talk [part 1] [abc].en-orig.srt").write_text("subs")
    download._rename_subtitles(video, "en")
    assert (tmp_path / "Talk [part 1] [abc].srt").read_text() == "subs"
    assert not (tmp_path / "Talk [part 1] [abc].en-orig.srt").exists()


def test_options_ask_for_srt_only_when_wanted(tmp_path: Path):
    plain = download.options(tmp_path, print, None, None)
    assert "writesubtitles" not in plain and plain["merge_output_format"] == "mp4"
    subs = download.options(tmp_path, print, None, "de")
    assert subs["subtitleslangs"] == ["de"] and subs["writeautomaticsub"]


def test_progress_hook_reports_fractions(tmp_path: Path):
    seen: list[float] = []
    hook = download.options(tmp_path, print, seen.append, None)["progress_hooks"][0]
    hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 200})
    hook({"status": "downloading", "downloaded_bytes": 10})  # no total yet: nothing to report
    hook({"status": "finished"})
    assert seen == [0.25]


def fake_yt_dlp(monkeypatch: pytest.MonkeyPatch, make_file: bool) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class YoutubeDL:
        def __init__(self, opts: dict[str, Any]) -> None:
            self.opts = opts
            calls.append(opts)

        def __enter__(self) -> "YoutubeDL":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, Any]:
            if make_file:
                (Path(self.opts["paths"]["home"]) / "Clip [id1].mp4").write_bytes(b"video")
                (Path(self.opts["paths"]["home"]) / "Clip [id1].en.srt").write_text("subs")
            return {"id": "id1", "title": "Clip", "ext": "webm"}

        def prepare_filename(self, info: dict[str, Any]) -> str:
            return str(Path(self.opts["paths"]["home"]) / "Clip [id1].webm")

    utils = types.SimpleNamespace(DownloadError=type("DownloadError", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=YoutubeDL, utils=utils))
    return calls


def test_download_returns_the_merged_mp4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = fake_yt_dlp(monkeypatch, make_file=True)
    video = download.download("https://x.test/v", tmp_path / "dl", log=lambda _m: None, subtitles="en")
    assert video == tmp_path / "dl" / "Clip [id1].mp4"
    assert (tmp_path / "dl" / "Clip [id1].srt").exists()
    assert calls[0]["noplaylist"] and calls[0]["updatetime"] is False


def test_download_that_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_yt_dlp(monkeypatch, make_file=False)
    with pytest.raises(download.DownloadError, match="isn't there"):
        download.download("https://x.test/v", tmp_path, log=lambda _m: None)


def test_cli_without_yt_dlp(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(download, "available", lambda: False)
    assert main(["https://www.youtube.com/watch?v=abc"]) == 2
    assert "clipwright[link]" in capsys.readouterr().err


def test_cli_checks_options_before_downloading(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    # CI installs no extras, so pretend yt-dlp is there to reach the option checks
    monkeypatch.setattr(download, "available", lambda: True)
    monkeypatch.setattr(download, "download", lambda *a, **k: pytest.fail("should not download"))
    assert main(["https://www.youtube.com/watch?v=abc", "--min", "30", "--max", "10"]) == 2
    assert "--max must be" in capsys.readouterr().err
