import numpy as np
import jax.numpy as jnp
import jaxlie
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.controller import IKController
from teleop_xr.messages import (
    XRButtonState,
    XRDeviceRole,
    XRGamepad,
    XRHandedness,
    XRInputSource,
    XRPose,
    XRState,
)


class DummyRobot(BaseRobot):
    def get_vis_config(self):
        return None

    @property
    def joint_var_cls(self):
        return None

    def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
        ident = jaxlie.SE3.identity()
        return {"left": ident, "right": ident, "head": ident}

    def get_default_config(self) -> jnp.ndarray:
        return jnp.array([0.0, 0.0])

    def build_costs(self, target_L, target_R, target_Head):
        return []


class DummySolver(PyrokiSolver):
    def __init__(self, out: np.ndarray):
        # We don't call super().__init__ because we don't want to trigger _warmup
        self.out = out

    def solve(self, target_L, target_R, target_Head, q_current):
        return jnp.asarray(self.out)


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
        robot=robot, solver=solver, filter_weights=np.array([0.5, 0.5])
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


def test_ik_controller_reset():
    robot = DummyRobot()
    controller = IKController(robot=robot)
    controller.active = True
    ident = jaxlie.SE3.identity()
    controller.snapshot_xr = {"left": ident}
    controller.reset()
    assert controller.active is False
    assert controller.snapshot_xr == {}
    assert controller.snapshot_robot == {}
