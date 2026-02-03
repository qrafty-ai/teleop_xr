import numpy as np
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


def test_camera_stream_track_with_external_source():
    source = ExternalVideoSource()
    track = CameraStreamTrack(source, "test_id")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    source.put_frame(frame)

    assert track.source == source
    assert track.stream_id == "test_id"


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
