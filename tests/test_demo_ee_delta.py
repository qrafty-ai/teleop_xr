import logging
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("rich")

from teleop_xr.demo.__main__ import run_right_ee_delta_demo


class DummyController:
    def __init__(self):
        self.mode = "teleop"
        self.mode_history = [self.mode]

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = str(mode)
        self.mode_history.append(self.mode)

    def submit_ee_delta(self, command, q_current):
        assert command["frame"] == "right"
        return np.array(q_current) + 1.0


class DummyRobot:
    actuated_joint_names = ["j1", "j2"]


class DummyTeleop:
    async def publish_joint_state(self, _joint_dict):
        return None


def test_run_right_ee_delta_demo_switches_mode_and_restores(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    controller = DummyController()
    robot = DummyRobot()
    teleop = DummyTeleop()
    q0 = np.array([0.0, 0.0])

    out = run_right_ee_delta_demo(
        controller=cast(Any, controller),
        robot=cast(Any, robot),
        teleop=cast(Any, teleop),
        q_current=q0,
        teleop_loop=None,
        logger=logging.getLogger("demo-test"),
    )

    np.testing.assert_allclose(out, [40.0, 40.0])
    assert controller.mode == "teleop"
    assert controller.mode_history[1] == "ee_delta"
    assert controller.mode_history[-1] == "teleop"
