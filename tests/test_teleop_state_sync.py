import pytest
import sys

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Fragile networking tests on Windows CI"
)

from unittest.mock import MagicMock, AsyncMock  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from teleop_xr import Teleop  # noqa: E402
from teleop_xr.config import TeleopSettings, RobotVisConfig  # noqa: E402


@pytest.mark.anyio
async def test_teleop_state_sync_on_connect():
    # 1. Setup Teleop with RobotVisConfig
    vis_config = RobotVisConfig(urdf_path="dummy.urdf")
    settings = TeleopSettings(robot_vis=vis_config)
    teleop = Teleop(settings)

    # Mock robot_vis module to prevent file loading issues and track calls
    teleop.robot_vis = MagicMock()
    teleop.robot_vis.get_frontend_config.return_value = {"urdf_url": "dummy"}
    teleop.robot_vis.broadcast_state = AsyncMock()

    # 2. Publish a state *before* any client connects
    initial_joints = {"joint1": 1.0, "joint2": 0.5}
    await teleop.publish_joint_state(initial_joints)

    # 3. Connect a client and verify it receives the state
    client = TestClient(teleop.app)

    with client.websocket_connect("/ws") as ws:
        # Expect config
        msg_config = ws.receive_json()
        assert msg_config["type"] == "config"

        # Expect robot_config
        msg_robot_config = ws.receive_json()
        assert msg_robot_config["type"] == "robot_config"

        # Expect robot_state (THIS IS WHAT WE ADDED)
        msg_state = ws.receive_json()
        assert msg_state["type"] == "robot_state"
        assert msg_state["data"]["joints"] == initial_joints


@pytest.mark.anyio
async def test_teleop_no_state_sync_if_not_published():
    # Setup without publishing state
    vis_config = RobotVisConfig(urdf_path="dummy.urdf")
    settings = TeleopSettings(robot_vis=vis_config)
    teleop = Teleop(settings)

    teleop.robot_vis = MagicMock()
    teleop.robot_vis.get_frontend_config.return_value = {"urdf_url": "dummy"}

    client = TestClient(teleop.app)

    with client.websocket_connect("/ws") as ws:
        msg_config = ws.receive_json()
        assert msg_config["type"] == "config"

        msg_robot_config = ws.receive_json()
        assert msg_robot_config["type"] == "robot_config"

        # Should NOT receive robot_state.
        # To verify, we can try to receive with a short timeout or just check there are no more messages?
        # TestClient doesn't support timeout easily on receive_json without blocking forever if empty.
        # But we can assume if we send something else, we get a response, and no state message in between.

        # Send a control check to verify connection is alive and we didn't miss messages
        ws.send_json({"type": "control_check", "client_id": "c1"})
        msg_control = ws.receive_json()
        assert msg_control["type"] == "control_status"
        # If robot_state was queued, it would have arrived before control_status
