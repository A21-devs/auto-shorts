from pathlib import Path

import pytest

from autoshorts.probe import MediaInfo
from autoshorts.render import RenderOptions, build_command, video_filter

INFO = MediaInfo(Path("in.mp4"), duration=600, width=1920, height=1080, fps=60, has_audio=True, video_stream=0)


@pytest.mark.parametrize("layout", ["crop", "blur", "fit"])
def test_every_layout_ends_in_vout(layout):
    g = video_filter(layout, None, 30, 60)
    assert g.startswith("[0:v]") and g.endswith("[vout]")


def test_fps_is_capped():
    assert "fps=60" in video_filter("crop", None, 120, 60)
    assert "fps=" not in video_filter("crop", None, 30, 60)


def test_subtitles_are_burned_last():
    g = video_filter("blur", "clip01.ass", 30, 60)
    assert g.endswith("ass=clip01.ass[vout]")


def test_command_seeks_and_maps_the_probed_stream():
    info = MediaInfo(Path("in.mp4"), 600, 1920, 1080, 30, True, video_stream=2)
    cmd = build_command(info, 12.5, 42.5, Path("out.mp4"), RenderOptions(encoder="libx264"))
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd[cmd.index("-t") + 1] == "30.000"
    assert cmd[cmd.index("-filter_complex") + 1].startswith("[0:2]")
    assert "libx264" in cmd and "loudnorm=I=-14:TP=-1.5:LRA=11" in cmd


def test_no_audio_means_no_audio_args():
    info = MediaInfo(Path("in.mp4"), 600, 1920, 1080, 30, False, 0)
    cmd = build_command(info, 0, 30, Path("o.mp4"), RenderOptions(encoder="libx264"))
    assert "-c:a" not in cmd and "0:a:0" not in cmd
