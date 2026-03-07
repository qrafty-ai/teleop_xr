from typing import cast

import pytest

try:
    import jaxlie  # noqa: F401
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip("IK dependencies not installed", allow_module_level=True)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
from unittest.mock import patch, PropertyMock  # noqa: E402
from loguru import logger  # noqa: E402
from teleop_xr.ik.robot import BaseRobot  # noqa: E402
from teleop_xr.ik.solver import PyrokiSolver  # noqa: E402
from teleop_xr.ik.controller import IKController  # noqa: E402
from teleop_xr.ik.commands import EEAbsoluteCommand, EEDeltaCommand  # noqa: E402
from teleop_xr.ik.control_mode import ControlMode  # noqa: E402
from teleop_xr.messages import (  # noqa: E402
    XRButtonState,
    XRDeviceRole,
    XRGamepad,
    XRHandedness,
    XRInputSource,
    XRPose,
    XRState,
)


class DummyRobot(BaseRobot):
    def _load_default_urdf(self):
        return None

    def get_vis_config(self):
        return None

    @property
    def actuated_joint_names(self) -> list[str]:
        return ["joint1", "joint2"]

    @property
    def joint_var_cls(self):
        return None

    def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
        ident = jaxlie.SE3.identity()
        return {"left": ident, "right": ident, "head": ident}

    def get_default_config(self) -> jnp.ndarray:
        return jnp.array([0.0, 0.0])

    def build_costs(self, target_L, target_R, target_Head, q_current=None):
        return []


class DummySolver:
    def __init__(self, out: np.ndarray):
        # We don't call super().__init__ because we don't want to trigger _warmup
        self.out = out

    def solve(self, target_L, target_R, target_Head, q_current):
        return jnp.asarray(self.out)


class MissingFrameRobot(DummyRobot):
    def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
        ident = jaxlie.SE3.identity()
        return {"left": ident, "head": ident}


def _as_solver(solver: DummySolver) -> PyrokiSolver:
    return cast(PyrokiSolver, cast(object, solver))


def _pose(x: float, y: float, z: float) -> XRPose:
    return XRPose(
        position={"x": x, "y": y, "z": z},
        orientation={"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    )


def _deadman_gamepad(pressed: bool) -> XRGamepad:
    buttons = [
        XRButtonState(pressed=False, touched=False, value=0.0),
        XRButtonState(pressed=pressed, touched=False, value=1.0 if pressed else 0.0),
    ]
    return XRGamepad(buttons=buttons, axes=[])


def test_ik_controller_deadman_and_step_transitions():
    robot = DummyRobot()
    solver = DummySolver(np.array([1.0, 2.0]))
    controller = IKController(
        robot=robot,
        solver=_as_solver(solver),
        filter_weights=np.array([0.5, 0.5]),
    )

    left = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    right = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    head = XRInputSource(role=XRDeviceRole.HEAD, pose=_pose(0.0, 0.2, 0.0))

    state = XRState(timestamp_unix_ms=1.0, devices=[left, right, head])
    q0 = np.array([0.0, 0.0])

    out0 = controller.step(state, q0)
    assert controller.active is True
    np.testing.assert_allclose(out0, q0)

    out1 = controller.step(state, q0)
    np.testing.assert_allclose(out1, [1.0, 2.0])

    left2 = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(False),
    )
    right2 = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(False),
    )
    state2 = XRState(timestamp_unix_ms=2.0, devices=[left2, right2, head])

    out2 = controller.step(state2, q0)
    assert controller.active is False
    np.testing.assert_allclose(out2, q0)
    assert controller.filter is not None
    assert controller.filter.data_ready() is False


def test_ik_controller_deadman_requires_two_buttons():
    robot = DummyRobot()
    controller = IKController(robot=robot)

    buttons = [XRButtonState(pressed=True, touched=False, value=1.0)]
    state = XRState(
        timestamp_unix_ms=1.0,
        devices=[
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.LEFT,
                gripPose=_pose(0.0, 0.0, 0.0),
                gamepad=XRGamepad(buttons=buttons, axes=[]),
            ),
            XRInputSource(
                role=XRDeviceRole.CONTROLLER,
                handedness=XRHandedness.RIGHT,
                gripPose=_pose(0.0, 0.0, 0.0),
                gamepad=XRGamepad(buttons=buttons, axes=[]),
            ),
        ],
    )

    q0 = np.array([0.0, 0.0])
    out = controller.step(state, q0)
    assert controller.active is False
    np.testing.assert_allclose(out, q0)


def test_ik_controller_no_solver():
    robot = DummyRobot()
    controller = IKController(robot=robot)

    left = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    right = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    head = XRInputSource(role=XRDeviceRole.HEAD, pose=_pose(0.0, 0.2, 0.0))
    state = XRState(timestamp_unix_ms=1.0, devices=[left, right, head])
    q0 = np.array([0.0, 0.0])

    controller.step(state, q0)
    assert controller.active

    out = controller.step(state, q0)
    np.testing.assert_allclose(out, q0)


def test_ik_controller_no_filter():
    robot = DummyRobot()
    solver = DummySolver(np.array([5.0, 6.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))

    left = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    right = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    head = XRInputSource(role=XRDeviceRole.HEAD, pose=_pose(0.0, 0.2, 0.0))
    state = XRState(timestamp_unix_ms=1.0, devices=[left, right, head])
    q0 = np.array([0.0, 0.0])

    controller.step(state, q0)
    out = controller.step(state, q0)
    np.testing.assert_allclose(out, [5.0, 6.0])


def test_ik_controller_reset_with_filter():
    robot = DummyRobot()
    controller = IKController(robot=robot, filter_weights=np.array([0.5, 0.5]))
    assert controller.filter is not None
    controller.filter.add_data(np.array([1.0, 1.0]))
    assert len(controller.filter._data_queue) > 0

    controller.reset()
    assert controller.active is False
    assert len(controller.filter._data_queue) == 0


def test_ik_controller_unsupported_frame_warning():
    robot = DummyRobot()
    controller = IKController(robot=robot)

    # DummyRobot only supports left/right/head by default in BaseRobot,
    # but DummyRobot FK returns all three.
    # Let's mock robot.supported_frames to be just {"left"}
    with patch.object(
        DummyRobot, "supported_frames", new_callable=PropertyMock
    ) as mock_frames:
        mock_frames.return_value = {"left"}

        # Create state with right controller
        right = XRInputSource(
            role=XRDeviceRole.CONTROLLER,
            handedness=XRHandedness.RIGHT,
            gripPose=_pose(0.1, 0.0, 0.0),
            gamepad=_deadman_gamepad(True),
        )
        state = XRState(timestamp_unix_ms=1.0, devices=[right])

        # Collect logs
        logs = []
        handler_id = logger.add(lambda msg: logs.append(msg), level="WARNING")
        try:
            controller._get_device_poses(state)
            assert any("not supported by robot" in str(m) for m in logs)
        finally:
            logger.remove(handler_id)


def test_ik_controller_step_noop_outside_teleop_mode():
    robot = DummyRobot()
    solver = DummySolver(np.array([7.0, 8.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_DELTA)

    left = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    right = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    state = XRState(timestamp_unix_ms=1.0, devices=[left, right])
    q0 = np.array([0.0, 0.0])

    out = controller.step(state, q0)
    np.testing.assert_allclose(out, q0)
    assert controller.active is False


def test_ik_controller_step_noop_outside_teleop_mode_ee_absolute():
    robot = DummyRobot()
    solver = DummySolver(np.array([7.0, 8.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_ABSOLUTE)

    left = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.LEFT,
        gripPose=_pose(0.0, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    right = XRInputSource(
        role=XRDeviceRole.CONTROLLER,
        handedness=XRHandedness.RIGHT,
        gripPose=_pose(0.1, 0.0, 0.0),
        gamepad=_deadman_gamepad(True),
    )
    state = XRState(timestamp_unix_ms=1.0, devices=[left, right])
    q0 = np.array([0.0, 0.0])

    out = controller.step(state, q0)
    np.testing.assert_allclose(out, q0)
    assert controller.active is False


def test_ik_controller_submit_ee_delta_requires_mode_switch():
    robot = DummyRobot()
    solver = DummySolver(np.array([2.0, 3.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    q0 = np.array([0.0, 0.0])
    command = EEDeltaCommand.model_validate(
        {
            "frame": "right",
            "delta_pose": {
                "position": {"x": 0.01, "y": 0.0, "z": 0.0},
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        }
    )

    with pytest.raises(RuntimeError):
        controller.submit_ee_delta(command, q0)

    controller.set_mode(ControlMode.EE_DELTA)
    out = controller.submit_ee_delta(command, q0)
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_ik_controller_submit_ee_delta_rejects_unsupported_frame():
    robot = DummyRobot()
    solver = DummySolver(np.array([1.0, 1.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_DELTA)

    with patch.object(
        DummyRobot, "supported_frames", new_callable=PropertyMock
    ) as mock_frames:
        mock_frames.return_value = {"left"}
        with pytest.raises(ValueError):
            controller.submit_ee_delta(
                {
                    "frame": "right",
                    "delta_pose": {
                        "position": {"x": 0.01, "y": 0.0, "z": 0.0},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                },
                np.array([0.0, 0.0]),
            )


def test_ik_controller_set_mode_noop_keeps_state():
    robot = DummyRobot()
    controller = IKController(robot=robot, filter_weights=np.array([0.5, 0.5]))
    assert controller.filter is not None

    controller.active = True
    ident = jaxlie.SE3.identity()
    controller.snapshot_xr = {"left": ident}
    controller.snapshot_robot = {"left": ident}
    controller.filter.add_data(np.array([0.1, 0.2]))

    controller.set_mode(ControlMode.TELEOP)

    assert controller.active is True
    assert "left" in controller.snapshot_xr
    assert "left" in controller.snapshot_robot
    assert len(controller.filter._data_queue) == 1


def test_ik_controller_set_mode_switch_resets_state_and_filter():
    robot = DummyRobot()
    controller = IKController(robot=robot, filter_weights=np.array([0.5, 0.5]))
    assert controller.filter is not None

    controller.active = True
    ident = jaxlie.SE3.identity()
    controller.snapshot_xr = {"left": ident}
    controller.snapshot_robot = {"left": ident}
    controller.filter.add_data(np.array([0.3, 0.4]))

    controller.set_mode(ControlMode.EE_DELTA)

    assert controller.get_mode() == ControlMode.EE_DELTA
    assert controller.active is False
    assert controller.snapshot_xr == {}
    assert controller.snapshot_robot == {}
    assert len(controller.filter._data_queue) == 0


def test_ik_controller_submit_ee_delta_no_solver_returns_current():
    robot = DummyRobot()
    controller = IKController(robot=robot)
    controller.set_mode(ControlMode.EE_DELTA)
    q0 = np.array([1.0, 2.0])

    out = controller.submit_ee_delta(
        {
            "frame": "right",
            "delta_pose": {
                "position": {"x": 0.01, "y": 0.0, "z": 0.0},
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        q0,
    )

    np.testing.assert_allclose(out, q0)


def test_ik_controller_submit_ee_delta_missing_fk_frame_raises():
    robot = MissingFrameRobot()
    solver = DummySolver(np.array([1.0, 2.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_DELTA)

    with pytest.raises(ValueError, match="missing from forward kinematics"):
        controller.submit_ee_delta(
            {
                "frame": "right",
                "delta_pose": {
                    "position": {"x": 0.01, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
            np.array([0.0, 0.0]),
        )


def test_ik_controller_submit_ee_absolute_requires_mode_switch():
    robot = DummyRobot()
    solver = DummySolver(np.array([2.0, 3.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    q0 = np.array([0.0, 0.0])
    command = EEAbsoluteCommand.model_validate(
        {
            "frame": "right",
            "target_pose": {
                "position": {"x": 0.3, "y": 0.1, "z": 0.2},
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        }
    )

    with pytest.raises(RuntimeError):
        controller.submit_ee_absolute(command, q0)

    controller.set_mode(ControlMode.EE_ABSOLUTE)
    out = controller.submit_ee_absolute(command, q0)
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_ik_controller_submit_ee_absolute_rejects_unsupported_frame():
    robot = DummyRobot()
    solver = DummySolver(np.array([1.0, 1.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_ABSOLUTE)

    with patch.object(
        DummyRobot, "supported_frames", new_callable=PropertyMock
    ) as mock_frames:
        mock_frames.return_value = {"left"}
        with pytest.raises(ValueError):
            controller.submit_ee_absolute(
                {
                    "frame": "right",
                    "target_pose": {
                        "position": {"x": 0.3, "y": 0.1, "z": 0.2},
                        "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                    },
                },
                np.array([0.0, 0.0]),
            )


def test_ik_controller_submit_ee_absolute_no_solver_returns_current():
    robot = DummyRobot()
    controller = IKController(robot=robot)
    controller.set_mode(ControlMode.EE_ABSOLUTE)
    q0 = np.array([1.0, 2.0])

    out = controller.submit_ee_absolute(
        {
            "frame": "right",
            "target_pose": {
                "position": {"x": 0.3, "y": 0.1, "z": 0.2},
                "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            },
        },
        q0,
    )

    np.testing.assert_allclose(out, q0)


def test_ik_controller_submit_ee_absolute_missing_fk_frame_raises():
    robot = MissingFrameRobot()
    solver = DummySolver(np.array([1.0, 2.0]))
    controller = IKController(robot=robot, solver=_as_solver(solver))
    controller.set_mode(ControlMode.EE_ABSOLUTE)

    with pytest.raises(ValueError, match="missing from forward kinematics"):
        controller.submit_ee_absolute(
            {
                "frame": "right",
                "target_pose": {
                    "position": {"x": 0.3, "y": 0.1, "z": 0.2},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
            np.array([0.0, 0.0]),
        )
