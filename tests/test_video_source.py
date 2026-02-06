import numpy as np
import logging
from unittest.mock import MagicMock, patch
from teleop_xr.video_stream import (
    OpenCVVideoSource,
    ExternalVideoSource,
    CameraStreamTrack,
)


def test_external_video_source():
    source = ExternalVideoSource()
    assert not source.grabbed
    assert not source.new_frame_event.is_set()

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source.put_frame(frame)

    assert source.grabbed
    assert source.new_frame_event.is_set()

    ok, read_frame = source.read()
    assert ok
    assert read_frame is not None
    assert np.array_equal(frame, read_frame)


def test_camera_stream_track_id():
    source = ExternalVideoSource()
    track = CameraStreamTrack(source, "test_id")
    assert track.id == "test_id"


def test_opencv_video_source_fail(caplog):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cap.read.return_value = (False, None)
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with caplog.at_level(logging.WARNING):
            source = OpenCVVideoSource(99, 640, 480, 30)
    assert "Failed to open video source" in caplog.text
    assert "Failed to read initial frame" in caplog.text
    assert not source.grabbed


def test_opencv_video_source_mock():
    # Mock cv2.VideoCapture
    mock_cap = MagicMock()
    with patch("cv2.VideoCapture", return_value=mock_cap):
        # Mock cap.read()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        source = OpenCVVideoSource(0, 640, 480, 30)

        assert source.grabbed
        ok, read_frame = source.read()
        assert ok
        assert read_frame is not None
        assert np.array_equal(frame, read_frame)

        source.start()
        assert source.started
        source.stop()
        assert not source.started
