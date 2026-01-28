from unittest.mock import Mock

from teleop.video_stream import route_video_message


def test_route_video_answer_calls_manager():
    manager = Mock()
    route_video_message(
        manager, {"type": "video_answer", "data": {"sdp": "s", "type": "answer"}}
    )
    manager.handle_answer.assert_called_once_with("s", "answer")


def test_route_video_ice_calls_manager():
    manager = Mock()
    route_video_message(manager, {"type": "video_ice", "data": {"candidate": "c"}})
    manager.add_ice.assert_called_once_with({"candidate": "c"})
