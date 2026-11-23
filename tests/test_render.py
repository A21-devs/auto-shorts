from pathlib import Path

import pytest

from autoshorts.probe import MediaInfo
from autoshorts.render import (
    RenderOptions,
    build_command,
    gain_chain,
    jumpcut_filter,
    parse_integrated,
    video_filter,
)

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
    assert "libx264" in cmd


LOUDNORM_OUT = """
[Parsed_loudnorm_0 @ 0000012]
{
	"input_i" : "-22.41",
	"input_tp" : "-6.02",
	"input_lra" : "4.30",
	"input_thresh" : "-32.80",
	"target_offset" : "0.10"
}
"""


def test_parse_integrated():
    assert parse_integrated(LOUDNORM_OUT) == -22.41
    assert parse_integrated(LOUDNORM_OUT.replace('"-22.41"', '"-inf"')) is None
    assert parse_integrated("no json here") is None


def test_gain_chain_limits_peaks():
    chain = gain_chain(6.5)
    assert chain.startswith("volume=6.50dB,alimiter=limit=0.841")


def test_command_uses_gain_chain_only_when_given():
    with_gain = build_command(INFO, 0, 30, Path("o.mp4"), RenderOptions(encoder="libx264"), gain_db=3.0)
    without = build_command(INFO, 0, 30, Path("o.mp4"), RenderOptions(encoder="libx264"))
    assert "volume=3.00dB" in with_gain[with_gain.index("-af") + 1]
    assert "-af" not in without

def test_no_audio_means_no_audio_args():
    info = MediaInfo(Path("in.mp4"), 600, 1920, 1080, 30, False, 0)
    cmd = build_command(info, 0, 30, Path("o.mp4"), RenderOptions(encoder="libx264"))
    assert "-c:a" not in cmd and "0:a:0" not in cmd


def test_jumpcut_filter_trims_and_joins_video_and_audio():
    g = jumpcut_filter([(0.0, 4.1), (4.9, 10.0)], audio=True, gain="volume=2.00dB")
    assert g.startswith("[0:v]split=2[s0][s1];[0:a:0]asplit=2[t0][t1];")
    assert "[s1]trim=start=4.900:end=10.000,setpts=PTS-STARTPTS[v1]" in g
    assert "[t1]atrim=start=4.900:end=10.000,asetpts=PTS-STARTPTS[a1]" in g
    assert g.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[jv][ja];[ja]volume=2.00dB[aout]")


def test_jumpcut_filter_without_audio():
    g = jumpcut_filter([(0.0, 1.0), (2.0, 3.0)], audio=False, gain=None)
    assert "asplit" not in g and g.endswith("[v0][v1]concat=n=2:v=1:a=0[jv]")


def test_command_with_cuts_maps_the_joined_audio_and_moves_the_gain_into_the_graph():
    info = MediaInfo(Path("in.mp4"), 600, 1920, 1080, 30, True, video_stream=2)
    cmd = build_command(info, 100, 110, Path("o.mp4"), RenderOptions(encoder="libx264"), gain_db=3.0,
                        keep=[(0.0, 4.1), (4.9, 10.0)])
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.startswith("[0:2]split=2")       # probed stream, not [0:v]
    assert "[jv]split[a][b]" in graph              # the layout reads the joined video
    assert "[0:v]" not in graph
    assert "volume=3.00dB" in graph and "-af" not in cmd
    assert cmd[cmd.index("[vout]") + 2] == "[aout]"


def test_keeping_everything_renders_straight_through():
    plain = build_command(INFO, 100, 110, Path("o.mp4"), RenderOptions(encoder="libx264"))
    full = build_command(INFO, 100, 110, Path("o.mp4"), RenderOptions(encoder="libx264"), keep=[(0.0, 10.0)])
    assert plain == full
