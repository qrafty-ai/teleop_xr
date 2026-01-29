import pytest
import logging
from teleop.camera_views import (
    parse_device_spec,
    build_camera_views_config,
    build_video_streams,
)


def test_parse_device_spec_valid():
    assert parse_device_spec(0) == 0
    assert parse_device_spec("1") == 1
    assert parse_device_spec("/dev/video0") == "/dev/video0"


def test_parse_device_spec_invalid():
    with pytest.raises(ValueError):
        parse_device_spec("")
    with pytest.raises(ValueError):
        parse_device_spec("   ")
    with pytest.raises(ValueError):
        parse_device_spec(None)
    with pytest.raises(ValueError):
        parse_device_spec(True)
    with pytest.raises(ValueError):
        parse_device_spec(False)
    with pytest.raises(ValueError):
        parse_device_spec("not_a_device")


def test_parse_device_spec_whitespace():
    assert parse_device_spec(" 2 ") == 2
    assert parse_device_spec(" /dev/video0  ") == "/dev/video0"


def test_build_camera_views_config_basic():
    config = build_camera_views_config(head=0, wrist_left="/dev/video1")
    assert config == {"head": {"device": 0}, "wrist_left": {"device": "/dev/video1"}}
    assert "wrist_right" not in config


def test_build_camera_views_config_duplicates(caplog):
    with caplog.at_level(logging.WARNING):
        config = build_camera_views_config(head=0, wrist_left=0)
    assert config == {"head": {"device": 0}, "wrist_left": {"device": 0}}
    assert "head" in caplog.text
    assert "wrist_left" in caplog.text
    assert "warning" in caplog.text.lower()


def test_build_video_streams_ordering():
    camera_views = {
        "wrist_right": {"device": 2},
        "head": {"device": 0},
        "wrist_left": {"device": 1},
    }
    streams = build_video_streams(camera_views)
    expected = {
        "streams": [
            {"id": "head", "device": 0},
            {"id": "wrist_left", "device": 1},
            {"id": "wrist_right", "device": 2},
        ]
    }
    assert streams == expected


def test_build_video_streams_partial():
    camera_views = {
        "wrist_left": {"device": "/dev/video1"},
        "head": {"device": 0},
    }
    streams = build_video_streams(camera_views)
    expected = {
        "streams": [
            {"id": "head", "device": 0},
            {"id": "wrist_left", "device": "/dev/video1"},
        ]
    }
    assert streams == expected


def test_build_camera_views_config_empty():
    assert build_camera_views_config() == {}


def test_build_video_streams_empty():
    assert build_video_streams({}) == {"streams": []}


def test_parse_device_spec_string_digits():
    assert parse_device_spec("123") == 123
