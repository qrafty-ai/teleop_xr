import pytest

from teleop.video_stream import parse_video_config


def test_parse_video_config_accepts_multiple_streams():
    payload = {
        "streams": [
            {
                "id": "cam0",
                "device": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "codec": "vp8",
                "bitrate_kbps": 1000,
            },
            {
                "id": "cam1",
                "device": 1,
                "width": 1280,
                "height": 720,
                "fps": 15,
                "codec": "vp8",
                "bitrate_kbps": 800,
            },
        ]
    }
    configs = parse_video_config(payload)
    assert [c.id for c in configs] == ["cam0", "cam1"]
    assert configs[0].width == 640


def test_parse_video_config_rejects_duplicate_ids():
    payload = {"streams": [{"id": "cam", "device": 0}, {"id": "cam", "device": 1}]}
    with pytest.raises(ValueError):
        parse_video_config(payload)
