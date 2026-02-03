import pytest
from unittest.mock import MagicMock, patch
from typing import cast
from teleop_xr.video_stream import (
    parse_video_config,
    VideoStreamManager,
    route_video_message,
    VideoStreamConfig,
    build_sources,
    VideoSource,
)


def test_parse_video_config():
    payload = {
        "streams": [
            {"id": "cam1", "device": 0, "width": 640, "height": 480},
            {"id": "cam2", "device": 1, "enabled": False},
        ]
    }
    configs = parse_video_config(payload)
    assert len(configs) == 2
    assert configs[0].id == "cam1"
    assert configs[0].width == 640
    assert configs[0].enabled is True
    assert configs[1].id == "cam2"
    assert configs[1].enabled is False


def test_parse_video_config_errors():
    with pytest.raises(ValueError, match="streams must be a list"):
        parse_video_config({"streams": "invalid"})

    with pytest.raises(ValueError, match="stream entries must be objects"):
        parse_video_config({"streams": ["invalid"]})

    with pytest.raises(ValueError, match="stream id missing or duplicate"):
        parse_video_config({"streams": [{"device": 0}]})


def test_build_sources():
    configs = [
        VideoStreamConfig(id="cam1", device=0, enabled=True),
        VideoStreamConfig(id="cam2", device=1, enabled=False),
    ]
    with patch("teleop_xr.video_stream.OpenCVVideoSource") as MockSource:
        sources = build_sources(configs)
        assert len(sources) == 1
        assert "cam1" in sources
        assert "cam2" not in sources
        MockSource.assert_called_once()


@pytest.mark.anyio
async def test_video_stream_manager():
    from unittest.mock import AsyncMock

    mock_pc = MagicMock()
    mock_pc.createOffer = AsyncMock(return_value=MagicMock())
    mock_pc.setLocalDescription = AsyncMock()
    mock_pc.setRemoteDescription = AsyncMock()
    mock_pc.addIceCandidate = AsyncMock()
    mock_pc.close = AsyncMock()

    with patch("teleop_xr.video_stream.RTCPeerConnection", return_value=mock_pc):
        sources = cast(dict[str, VideoSource], {"cam1": MagicMock()})
        manager = VideoStreamManager(sources)

        await manager.create_offer()

        mock_pc.addTrack.assert_called()
        mock_pc.createOffer.assert_called()
        mock_pc.setLocalDescription.assert_called()

        await manager.handle_answer("sdp", "answer")
        mock_pc.setRemoteDescription.assert_called()

        await manager.add_ice(
            {
                "candidate": "candidate:842163049 1 udp 1677729535 192.168.1.2 56000 typ host generation 0",
                "sdpMid": "0",
                "sdpMLineIndex": 0,
            }
        )
        mock_pc.addIceCandidate.assert_called()

        await manager.close()
        mock_pc.close.assert_called()


@pytest.mark.anyio
async def test_route_video_message():
    from unittest.mock import AsyncMock

    manager = MagicMock()
    manager.handle_answer = AsyncMock()
    manager.add_ice = AsyncMock()

    await route_video_message(manager, {"type": "video_answer", "data": {"sdp": "sdp"}})
    manager.handle_answer.assert_called_with("sdp", "answer")

    await route_video_message(
        manager, {"type": "video_ice", "data": {"candidate": "..."}}
    )
    manager.add_ice.assert_called()
