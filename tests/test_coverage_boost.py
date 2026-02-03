import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from teleop_xr import Teleop, TeleopSettings
from typing import Any
from unittest.mock import MagicMock


def test_missing_client_id_handling():
    teleop: Any = Teleop(TeleopSettings())
    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"type": "control_check", "data": {}})
        resp = ws.receive_json()
        assert resp["type"] == "control_status"
        assert resp["data"]["in_control"] is False

        ws.send_json({"type": "xr_state", "data": {"devices": []}})
        resp = ws.receive_json()
        assert resp["type"] == "deny"
        assert resp["data"]["reason"] == "missing_client_id"

        ws.send_json({"type": "video_request", "data": {}})
        resp = ws.receive_json()
        assert resp["type"] == "deny"
        assert resp["data"]["reason"] == "missing_client_id"


def test_console_log_filtering(caplog):
    teleop: Any = Teleop(TeleopSettings())
    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json(
            {
                "type": "console_log",
                "client_id": "c1",
                "data": {"level": "info", "message": "hello world"},
            }
        )

        ws.send_json(
            {
                "type": "console_log",
                "client_id": "c1",
                "data": {
                    "level": "error",
                    "message": "WS Error: [object Event] isTrusted: true",
                },
            }
        )

        time.sleep(0.1)

    assert "hello world" in caplog.text
    assert "WS Error" not in caplog.text


@pytest.mark.anyio
async def test_connection_manager_broken_broadcast():
    from teleop_xr import ConnectionManager

    manager = ConnectionManager()

    mock_ws_ok = MagicMock()

    async def mock_send_ok(msg):
        pass

    mock_ws_ok.send_text = mock_send_ok

    mock_ws_broken = MagicMock()

    async def mock_send_fail(msg):
        raise Exception("Broken")

    mock_ws_broken.send_text = mock_send_fail

    await manager.register(mock_ws_ok)
    await manager.register(mock_ws_broken)

    assert len(manager.active_connections) == 2

    await manager.broadcast("test message")

    assert len(manager.active_connections) == 1
    assert mock_ws_ok in manager.active_connections


def test_controller_expiry_releases_video(monkeypatch):
    t = {"now": 0.0}

    def monotonic() -> float:
        return t["now"]

    import teleop_xr

    monkeypatch.setattr(teleop_xr.time, "monotonic", monotonic)

    settings = TeleopSettings()
    teleop: Any = Teleop(settings)

    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        ws1.receive_json()
        ws2.receive_json()

        ws1.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws1.receive_json()

        t["now"] += 6.0

        ws2.send_json({"type": "control_check", "client_id": "c2", "data": {}})
        ws2.receive_json()

        assert teleop._Teleop__controller_client_id == "c2"


@pytest.mark.anyio
async def test_robot_vis_broadcast_coverage():
    from teleop_xr.robot_vis import RobotVisModule

    mock_robot_vis = MagicMock(spec=RobotVisModule)

    settings = TeleopSettings()
    teleop: Any = Teleop(settings)
    teleop.robot_vis = mock_robot_vis

    await teleop.publish_joint_state({"j1": 0.0})

    mock_robot_vis.broadcast_state.assert_called_once()


def test_allowed_xr_state():
    teleop: Any = Teleop(TeleopSettings())
    teleop._Teleop__handle_xr_state = MagicMock()

    client = TestClient(teleop._Teleop__app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws.receive_json()

        ws.send_json({"type": "xr_state", "client_id": "c1", "data": {"devices": []}})

        time.sleep(0.1)

        teleop._Teleop__handle_xr_state.assert_called_once()


@pytest.mark.anyio
async def test_video_sessions_cleanup_logic(monkeypatch):
    t = {"now": 0.0}

    def monotonic() -> float:
        return t["now"]

    import teleop_xr

    monkeypatch.setattr(teleop_xr.time, "monotonic", monotonic)

    teleop: Any = Teleop(TeleopSettings())

    ws1 = MagicMock()
    ws2 = MagicMock()

    async def mock_close():
        pass

    sess1 = MagicMock()
    sess1.close = mock_close

    sess2 = MagicMock()
    sess2.close = mock_close

    teleop._Teleop__ws_client_ids = {ws1: "c1", ws2: "c2"}
    teleop._Teleop__video_sessions = {ws1: sess1, ws2: sess2}

    client = TestClient(teleop._Teleop__app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws.receive_json()

        assert ws1 in teleop._Teleop__video_sessions
        assert ws2 not in teleop._Teleop__video_sessions

        teleop._Teleop__video_sessions[ws2] = sess2

        t["now"] += 6.0
        ws.send_json({"type": "control_check", "client_id": "c2", "data": {}})
        ws.receive_json()

        assert ws1 not in teleop._Teleop__video_sessions
        assert ws2 in teleop._Teleop__video_sessions


def test_controller_disconnect_releases_control():
    teleop: Any = Teleop(TeleopSettings())
    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws.receive_json()
        assert teleop._Teleop__controller_client_id == "c1"

        ws.close()

    assert teleop._Teleop__controller_client_id is None


@pytest.mark.anyio
async def test_video_messages_allowed():
    teleop: Any = Teleop(TeleopSettings())

    async def mock_handle(ws, msg):
        pass

    teleop._handle_video_message = mock_handle

    client = TestClient(teleop._Teleop__app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()

        ws.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws.receive_json()

        mock_handler = MagicMock()

        async def async_mock_handler(*args, **kwargs):
            mock_handler(*args, **kwargs)

        teleop._handle_video_message = async_mock_handler

        for msg_type in ["video_request", "video_answer", "video_ice"]:
            ws.send_json({"type": msg_type, "client_id": "c1", "data": {}})

        await asyncio.sleep(0.1)
        assert mock_handler.call_count == 3


def test_video_messages_not_allowed():
    teleop: Any = Teleop(TeleopSettings())
    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        ws1.receive_json()
        ws2.receive_json()

        ws1.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        ws1.receive_json()

        for msg_type in ["video_request", "video_answer", "video_ice"]:
            ws2.send_json({"type": msg_type, "client_id": "c2", "data": {}})
            resp = ws2.receive_json()
            assert resp["type"] == "deny"
            assert resp["data"]["reason"] == "not_in_control"
