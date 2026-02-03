from typing import Any

from fastapi.testclient import TestClient

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings


def test_video_offer_only_for_controller_client_id():
    teleop: Any = Teleop(TeleopSettings())
    client = TestClient(teleop._Teleop__app)

    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        assert ws1.receive_json()["type"] == "config"
        assert ws2.receive_json()["type"] == "config"

        ws1.send_json({"type": "control_check", "client_id": "c1", "data": {}})
        status1 = ws1.receive_json()
        assert status1["type"] == "control_status"
        assert status1["data"]["in_control"] is True

        ws2.send_json({"type": "video_request", "client_id": "c2", "data": {}})
        deny = ws2.receive_json()
        assert deny["type"] == "deny"
        assert deny["data"]["reason"] == "not_in_control"
        assert deny["data"]["controller_client_id"] == "c1"

        ws1.send_json({"type": "video_request", "client_id": "c1", "data": {}})
        offer = ws1.receive_json()
        assert offer["type"] == "video_offer"
