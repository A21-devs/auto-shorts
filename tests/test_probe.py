from autoshorts.probe import _fps, _video_rank


def test_real_video_beats_bigger_thumbnail():
    video = {"index": 0, "codec_name": "h264", "width": 912, "height": 608, "nb_frames": "23880"}
    thumb = {"index": 2, "codec_name": "mjpeg", "width": 1826, "height": 1226, "nb_frames": "1"}
    assert max([video, thumb], key=_video_rank) is video


def test_unknown_frame_count_is_not_a_picture():
    # fragmented MP4: ffprobe reports nb_frames=0 for the real video
    video = {"codec_name": "h264", "width": 1280, "height": 720, "nb_frames": "0"}
    thumb = {"codec_name": "mjpeg", "width": 1920, "height": 1080, "nb_frames": "1"}
    assert max([video, thumb], key=_video_rank) is video


def test_attached_pic_is_skipped_even_if_h264():
    cover = {"codec_name": "h264", "width": 3000, "height": 3000, "disposition": {"attached_pic": 1}}
    video = {"codec_name": "h264", "width": 1280, "height": 720}
    assert max([cover, video], key=_video_rank) is video


def test_bigger_wins_between_real_videos():
    small = {"codec_name": "h264", "width": 640, "height": 360}
    big = {"codec_name": "hevc", "width": 1920, "height": 1080}
    assert max([small, big], key=_video_rank) is big


def test_fps_parsing():
    assert _fps("30000/1001") == 30000 / 1001
    assert _fps("60/1") == 60
    assert _fps("0/0") == 30.0
    assert _fps("garbage") == 30.0
