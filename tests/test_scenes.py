from clipwright.scenes import parse_showinfo


def test_parse_showinfo():
    stderr = (
        "[Parsed_showinfo_3 @ 000001] n:   0 pts:  12345 pts_time:2.469   duration:1\n"
        "frame=  10 fps=0.0 q=-0.0 size=N/A\n"
        "[Parsed_showinfo_3 @ 000001] n:   1 pts:  55000 pts_time:11     duration:1\n"
    )
    assert parse_showinfo(stderr) == [2.469, 11.0]


def test_parse_showinfo_ignores_other_lines():
    assert parse_showinfo("Input #0, mov,mp4\n  Duration: 00:14:00.00, pts_time:99\n") == []
