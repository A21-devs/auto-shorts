"""`clipwright FOLDER`, --json, and the post .txt written next to clips with --copy."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from conftest import make_video

from clipwright import cli, copywriting, pipeline
from clipwright.copywriting import PostCopy


def test_post_text_is_ready_to_paste():
    post = {"title": "He beat the boss", "description": "Two hours in.", "hashtags": ["gaming", "eldenring"]}
    assert pipeline.post_text(post) == "He beat the boss\n\nTwo hours in.\n\n#gaming #eldenring\n"
    assert pipeline.post_text({"title": "Only a title", "description": "", "hashtags": []}) == "Only a title\n"


def test_stale_post_files_are_removed(tmp_path: Path):
    for name in ["v_clip01.txt", "v_clip03.txt", "notes.txt"]:
        (tmp_path / name).write_text("x")
    assert pipeline.remove_stale_clips(tmp_path, "v", keep=2) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt", "v_clip01.txt"]


def fake_run(calls: list[Path], fail: set[str] = frozenset()):  # type: ignore[assignment]
    def run(job: pipeline.Job, log: Any = print, progress: Any = None) -> Path:
        calls.append(job.source)
        if job.source.name in fail:
            raise RuntimeError("no clips found, is the video shorter than --min?")
        assert job.out_dir is not None
        job.out_dir.mkdir(parents=True, exist_ok=True)
        manifest = job.out_dir / "manifest.json"
        manifest.write_text(json.dumps({"source": str(job.source.resolve()), "clips": []}), encoding="utf-8")
        return manifest
    return run


def test_folder_clips_every_video_and_skips_done_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]):
    for name in ["a.mp4", "b.mkv", "c.mp4", "notes.txt", "d.mp4.part"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.mp4").write_bytes(b"x")
    calls: list[Path] = []
    monkeypatch.setattr(cli, "run", fake_run(calls, fail={"b.mkv"}))
    # c already has clips from an earlier run
    done = tmp_path / "c_shorts"
    done.mkdir()
    (done / "manifest.json").write_text(json.dumps({"source": str((tmp_path / "c.mp4").resolve())}))

    assert cli.main([str(tmp_path), "--no-captions", "--json"]) == 1  # b failed
    assert [p.name for p in calls] == ["a.mp4", "b.mkv"]
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert [Path(d["video"]).name for d in result["done"]] == ["a.mp4"]
    assert result["skipped"] == 1 and result["failed"] == 1
    assert "failed: b.mkv: no clips found" in out.err and "c.mp4: already has clips" in out.err


def test_folder_rejects_one_srt(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "a.mp4").write_bytes(b"x")
    srt = tmp_path / "a.srt"
    srt.write_text("")
    assert cli.main([str(tmp_path), "--srt", str(srt)]) == 2
    assert "--srt is for one video" in capsys.readouterr().err


def test_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cli.main([str(tmp_path)]) == 2
    assert "no videos in" in capsys.readouterr().err


@pytest.mark.ffmpeg
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_json_and_post_file_on_a_real_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]):
    src = tmp_path / "long.mp4"
    make_video(src)
    (tmp_path / "long.srt").write_text("\n".join(
        f"{i + 1}\n00:00:{i * 5:02d},000 --> 00:00:{i * 5 + 4:02d},000\nline {i}.\n" for i in range(12)),
        encoding="utf-8")
    monkeypatch.setattr(copywriting, "write_copy",
                        lambda *a, **k: PostCopy("A title", "A description.", ["one", "two"]))
    assert cli.main([str(src), "-n", "1", "--min", "15", "--max", "20", "--encoder", "libx264",
                     "--copy", "claude", "--hook", "--json"]) == 0
    out = capsys.readouterr()
    data = json.loads(out.out)  # stdout is only the manifest
    clip = data["clips"][0]
    assert clip["post"] == "long_clip01.txt" and clip["hook"] == "A title"
    folder = tmp_path / "long_shorts"
    assert (folder / "long_clip01.txt").read_text(encoding="utf-8").startswith("A title\n\nA description.")
    assert "rendered long_clip01.mp4" in out.err


def test_json_keeps_the_download_log_off_stdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]):
    from clipwright import download

    video = tmp_path / "dl.mp4"
    video.write_bytes(b"x")

    def fake_download(url: str, folder: Path, log: Any = print, **_k: Any) -> Path:
        log("downloading 50%")
        return video

    monkeypatch.setattr(download, "available", lambda: True)
    monkeypatch.setattr(download, "download", fake_download)
    monkeypatch.setattr(cli, "run", fake_run([]))
    assert cli.main(["https://www.youtube.com/watch?v=x", "--no-captions", "--json", "-o", str(tmp_path / "o")]) == 0
    out = capsys.readouterr()
    json.loads(out.out)
    assert "downloading 50%" in out.err
