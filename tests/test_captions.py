from clipwright.captions import CaptionStyle, _ass_color, _ts, build_ass, build_srt, group_words
from clipwright.transcribe import Word


def words(*items: tuple[float, float, str]) -> list[Word]:
    return [Word(a, b, t) for a, b, t in items]


def test_timestamp_format():
    assert _ts(0) == "0:00:00.00"
    assert _ts(61.257) == "0:01:01.26"
    assert _ts(3725.5) == "1:02:05.50"
    assert _ts(-1) == "0:00:00.00"


def test_ass_color_is_bgr():
    assert _ass_color("FFD84D") == "&H004DD8FF"


def test_groups_split_on_count_sentence_and_pause():
    ws = words((0, 0.3, "one"), (0.3, 0.6, "two"), (0.6, 0.9, "three"), (0.9, 1.2, "four."),
               (1.2, 1.5, "five"), (3.0, 3.3, "six"))
    groups = group_words(ws, max_words=3, max_chars=40)
    assert [[w.text for w in g] for g in groups] == [["one", "two", "three"], ["four."], ["five"], ["six"]]


def test_groups_split_on_length():
    ws = words((0, 1, "supercalifragilistic"), (1, 2, "expialidocious"))
    assert len(group_words(ws, max_words=3, max_chars=18)) == 2


def test_build_ass_is_relative_to_clip_and_highlights():
    ws = words((9.0, 9.5, "before"), (10.2, 10.6, "hello"), (10.6, 11.0, "world"), (40.0, 41.0, "after"))
    doc = build_ass(ws, clip_start=10.0, clip_end=30.0, style=CaptionStyle())
    events = [line for line in doc.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 2
    assert events[0].startswith("Dialogue: 0,0:00:00.20,0:00:00.60,Cap")
    assert "{\\c&H004DD8FF}HELLO{\\r} WORLD" in events[0]
    assert "HELLO {\\c&H004DD8FF}WORLD{\\r}" in events[1]
    assert "BEFORE" not in doc and "AFTER" not in doc


def test_no_blank_flash_between_lines_but_real_pauses_clear():
    ws = words((0.0, 0.3, "one"), (0.3, 0.6, "two"), (0.6, 0.9, "three"),   # line 1
               (1.05, 1.4, "four"),                                         # 150 ms gap: hold
               (3.0, 3.4, "five"))                                          # 1.6 s pause: clear
    events = [line.split(",")[1:3] for line in build_ass(ws, 0, 10).splitlines() if line.startswith("Dialogue:")]
    starts_ends = [(s, e) for s, e in events]
    assert starts_ends[2] == ("0:00:00.60", "0:00:01.05")   # "three" held until "four" starts
    assert starts_ends[3] == ("0:00:01.05", "0:00:01.40")   # "four" not held across the long pause


def test_braces_in_words_cant_inject_ass_tags():
    doc = build_ass(words((0, 1, "{\\pos(0,0)}hi")), 0, 5)
    assert "{\\pos" not in doc


def test_no_words_gives_valid_header_only():
    doc = build_ass([], 0, 10)
    assert "[Events]" in doc and "Dialogue:" not in doc


def test_srt_uses_full_lines_relative_to_the_clip():
    words = [Word(9.0, 9.5, "before"), Word(10.2, 10.5, "so"), Word(10.5, 11.0, "this"), Word(11.0, 11.4, "is"),
             Word(11.4, 12.0, "the"), Word(12.0, 12.6, "part."), Word(14.0, 14.5, "Next"), Word(14.5, 15.2, "one")]
    srt = build_srt(words, 10.0, 15.0)
    assert srt == ("1\n00:00:00,200 --> 00:00:02,600\nso this is the part.\n\n"
                   "2\n00:00:04,000 --> 00:00:05,000\nNext one\n")


def test_srt_breaks_long_lines():
    words = [Word(i * 0.3, i * 0.3 + 0.25, "word") for i in range(20)]
    lines = [ln for ln in build_srt(words, 0.0, 10.0).splitlines() if ln.startswith("word")]
    assert len(lines) > 1 and all(len(ln) <= 42 for ln in lines)
