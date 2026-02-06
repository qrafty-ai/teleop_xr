import numpy as np
import jax.numpy as jnp
import jaxlie
from loguru import logger
from unittest.mock import MagicMock
from teleop_xr.ik.controller import IKController
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.messages import (
    XRState,
    XRInputSource,
    XRDeviceRole,
    XRHandedness,
    XRPose,
    XRGamepad,
    XRButtonState,
)


class MockRobot(BaseRobot):
    def __init__(self, supported_frames={"left", "right", "head"}):
        self._supported_frames = supported_frames

    @property
    def supported_frames(self):
        return self._supported_frames

    @property
    def actuated_joint_names(self):
        return [f"joint_{i}" for i in range(10)]

    def get_vis_config(self):
        return None

    @property
    def joint_var_cls(self):
        return MagicMock()

    def forward_kinematics(self, config):
        return {k: jaxlie.SE3.identity() for k in self._supported_frames}

    def get_default_config(self):
        return jnp.zeros(10)

    def build_costs(self, target_L, target_R, target_Head, q_current=None):
        return []


def create_xr_state(has_left=True, has_right=True, has_head=True, pressed=True):
    devices = []
    if has_head:
        devices.append(
            XRInputSource(
                role=XRDeviceRole.HEAD,
                pose=XRPose(
                    position={"x": 0, "y": 0, "z": 0},
                    orientation={"w": 1, "x": 0, "y": 0, "z": 0},
                ),
            )
        )
    if has_left:
        devices.append(
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.LEFT,
                gripPose=XRPose(
                    position={"x": 0, "y": 0, "z": 0},
                    orientation={"w": 1, "x": 0, "y": 0, "z": 0},
                ),
                gamepad=XRGamepad(
                    buttons=[
                        XRButtonState(pressed=False, touched=False, value=0),
                        XRButtonState(pressed=pressed, touched=pressed, value=1.0),
                    ],
                    axes=[],
                ),
            )
        )
    if has_right:
        devices.append(
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.RIGHT,
                gripPose=XRPose(
                    position={"x": 0, "y": 0, "z": 0},
                    orientation={"w": 1, "x": 0, "y": 0, "z": 0},
                ),
                gamepad=XRGamepad(
                    buttons=[
                        XRButtonState(pressed=False, touched=False, value=0),
                        XRButtonState(pressed=pressed, touched=pressed, value=1.0),
                    ],
                    axes=[],
                ),
            )
        )
    return XRState(timestamp_unix_ms=0, devices=devices)


def test_controller_supported_frames_subset():
    robot = MockRobot(supported_frames={"left", "right"})
    solver = MagicMock()
    solver.solve.return_value = jnp.zeros(10)
    controller = IKController(robot, solver)

    state = create_xr_state(has_left=True, has_right=True, has_head=True)
    config = np.zeros(10)

    controller.step(state, config)
    assert controller.active
    assert "head" not in controller.snapshot_xr
    assert "head" not in controller.snapshot_robot

    controller.step(state, config)

    args, kwargs = solver.solve.call_args
    assert args[0] is not None
    assert args[1] is not None
    assert args[2] is None


def test_controller_missing_required_frame():
    robot = MockRobot(supported_frames={"left", "right", "head"})
    controller = IKController(robot)

    state = create_xr_state(has_left=True, has_right=True, has_head=False)
    config = np.zeros(10)

    controller.step(state, config)
    assert not controller.active


def test_warning_unsupported_frame():
    robot = MockRobot(supported_frames={"left", "right"})
    controller = IKController(robot)

    state = create_xr_state(has_left=True, has_right=True, has_head=True)
    config = np.zeros(10)

    # Use a list to collect log messages
    logs = []
    handler_id = logger.add(lambda msg: logs.append(msg), level="WARNING")

    try:
        controller.step(state, config)
        assert any(
            "Warning: Frame 'head' is available in XRState but not supported by robot"
            in str(m)
            for m in logs
        )

        logs.clear()
        controller.step(state, config)
        assert not any("Warning" in str(m) for m in logs)
    finally:
        logger.remove(handler_id)
