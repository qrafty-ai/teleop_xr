from typing import Any
from unittest.mock import AsyncMock

import pytest

import teleop_xr
from teleop_xr.config import TeleopSettings


@pytest.mark.anyio
async def test_start_video_session_replaces_existing_manager(
    monkeypatch: pytest.MonkeyPatch,
):
    teleop: Any = teleop_xr.Teleop(TeleopSettings())
    websocket = object()
    closed: list[str] = []

    class ExistingSession:
        async def close(self) -> None:
            closed.append("closed")

    class ReplacementSession:
        def __init__(self, _sources: dict[str, Any]):
            self.closed = False

        async def create_offer(self):
            return type("Offer", (), {"sdp": "offer-sdp", "type": "offer"})()

    send_personal_message = AsyncMock()
    teleop._Teleop__manager.send_personal_message = send_personal_message
    teleop._Teleop__video_sessions[websocket] = ExistingSession()

    monkeypatch.setattr(teleop_xr, "VideoStreamManager", ReplacementSession)

    await teleop._start_video_session(websocket)

    assert closed == ["closed"]
    assert isinstance(teleop._Teleop__video_sessions[websocket], ReplacementSession)
    send_personal_message.assert_awaited_once()
