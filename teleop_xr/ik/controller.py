import jax.numpy as jnp
import jaxlie
import numpy as np
from loguru import logger
from teleop_xr.utils.filter import WeightedMovingFilter
from teleop_xr.messages import XRState, XRDeviceRole, XRHandedness, XRPose
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.solver import PyrokiSolver


class IKController:
    """
    High-level controller for teleoperation using IK.

    This class manages the transition between idle and active teleoperation,
    handles XR device snapshots for relative motion, and coordinates between
    the robot model, IK solver, and optional output filtering.
    """

    robot: BaseRobot
    solver: PyrokiSolver | None
    active: bool
    snapshot_xr: dict[str, jaxlie.SE3]
    snapshot_robot: dict[str, jaxlie.SE3]
    filter: WeightedMovingFilter | None
    _warned_unsupported: set[str]

    def __init__(
        self,
        robot: BaseRobot,
        solver: PyrokiSolver | None = None,
        filter_weights: np.ndarray | None = None,
    ):
        """
        Initialize the IK controller.

        Args:
            robot: The robot model.
            solver: The IK solver. If None, step() will return current_config.
            filter_weights: Optional weights for a WeightedMovingFilter on joint outputs.
        """
        self.robot = robot
        self.solver = solver
        self.active = False
        self._warned_unsupported = set()

        # Snapshots
        self.snapshot_xr = {}
        self.snapshot_robot = {}

        # Filter for joint configuration
        self.filter = None
        if filter_weights is not None:
            # We'll initialize the filter when we know the data size (from default config)
            default_config = self.robot.get_default_config()
            self.filter = WeightedMovingFilter(
                filter_weights, data_size=len(default_config)
            )

    def xr_pose_to_se3(self, pose: XRPose) -> jaxlie.SE3:
        """
        Convert an XRPose to a jaxlie SE3 object.

        Args:
            pose: The XR pose to convert.

        Returns:
            jaxlie.SE3: The converted pose.
        """
        translation = jnp.array(
            [pose.position["x"], pose.position["y"], pose.position["z"]]
        )
        rotation = jaxlie.SO3(
            wxyz=jnp.array(
                [
                    pose.orientation["w"],
                    pose.orientation["x"],
                    pose.orientation["y"],
                    pose.orientation["z"],
                ]
            )
        )
        return jaxlie.SE3.from_rotation_and_translation(rotation, translation)

    def compute_teleop_transform(
        self, t_ctrl_curr: jaxlie.SE3, t_ctrl_init: jaxlie.SE3, t_ee_init: jaxlie.SE3
    ) -> jaxlie.SE3:
        """
        Compute the target robot pose based on XR controller motion.

        Args:
            t_ctrl_curr: Current XR controller pose.
            t_ctrl_init: XR controller pose at the start of teleoperation.
            t_ee_init: Robot end-effector pose at the start of teleoperation.

        Returns:
            jaxlie.SE3: The calculated target pose for the robot end-effector.
        """
        t_delta_ros = t_ctrl_curr.translation() - t_ctrl_init.translation()
        t_delta_robot = self.robot.ros_to_base @ t_delta_ros

        q_delta_ros = t_ctrl_curr.rotation() @ t_ctrl_init.rotation().inverse()
        q_delta_robot = self.robot.ros_to_base @ q_delta_ros @ self.robot.base_to_ros

        t_new = t_ee_init.translation() + t_delta_robot
        q_new = q_delta_robot @ t_ee_init.rotation()

        return jaxlie.SE3.from_rotation_and_translation(q_new, t_new)

    def _get_device_poses(self, state: XRState) -> dict[str, jaxlie.SE3]:
        """
        Extract poses for relevant XR devices from the current state.
        """
        poses = {}
        supported = self.robot.supported_frames
        for device in state.devices:
            frame_name = None
            pose_data = None
            if device.role == XRDeviceRole.CONTROLLER:
                if device.handedness == XRHandedness.LEFT and device.gripPose:
                    frame_name = "left"
                    pose_data = device.gripPose
                elif device.handedness == XRHandedness.RIGHT and device.gripPose:
                    frame_name = "right"
                    pose_data = device.gripPose
            elif device.role == XRDeviceRole.HEAD and device.pose:
                frame_name = "head"
                pose_data = device.pose

            if frame_name and pose_data is not None:
                if frame_name in supported:
                    poses[frame_name] = self.xr_pose_to_se3(pose_data)
                elif frame_name not in self._warned_unsupported:
                    logger.warning(
                        f"[IKController] Warning: Frame '{frame_name}' is available in XRState but not supported by robot. Skipping."
                    )
                    self._warned_unsupported.add(frame_name)
        return poses

    def _check_deadman(self, state: XRState) -> bool:
        """
        Check if the deadman switch (usually trigger or grip) is engaged on both controllers.
        """
        left_squeezed = False
        right_squeezed = False
        for device in state.devices:
            if device.role == XRDeviceRole.CONTROLLER:
                is_squeezed = (
                    device.gamepad is not None
                    and len(device.gamepad.buttons) > 1
                    and device.gamepad.buttons[1].pressed
                )
                if device.handedness == XRHandedness.LEFT:
                    left_squeezed = is_squeezed
                elif device.handedness == XRHandedness.RIGHT:
                    right_squeezed = is_squeezed
        return left_squeezed and right_squeezed

    def reset(self) -> None:
        """
        Resets the controller state, forcing it to re-take snapshots on the next step.
        """
        self.active = False
        self.snapshot_xr = {}
        self.snapshot_robot = {}
        if self.filter is not None:
            self.filter.reset()
        logger.info("[IKController] Reset triggered")

    def step(self, state: XRState, q_current: np.ndarray) -> np.ndarray:
        """
        Execute one control step: update targets and solve for new joint configuration.

        Args:
            state: The current XR state from the headset.
            q_current: The current joint configuration of the robot.

        Returns:
            np.ndarray: The new (possibly filtered) joint configuration.
        """
        is_deadman_active = self._check_deadman(state)
        curr_xr_poses = self._get_device_poses(state)

        # Check if we have all necessary poses
        required_keys = self.robot.supported_frames
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        if is_deadman_active and has_all_poses:
            if not self.active:
                # Engagement transition: take snapshots
                self.active = True
                self.snapshot_xr = curr_xr_poses

                # Get initial robot FK poses
                # Cast q_current to jnp.ndarray for JAX-based robot models
                fk_poses = self.robot.forward_kinematics(jnp.asarray(q_current))
                self.snapshot_robot = {k: fk_poses[k] for k in required_keys}

                logger.info(f"[IKController] Initial Robot FK: {self.snapshot_robot}")
                return q_current

            # Active control
            target_L: jaxlie.SE3 | None = None
            target_R: jaxlie.SE3 | None = None
            target_Head: jaxlie.SE3 | None = None

            if "left" in required_keys:
                target_L = self.compute_teleop_transform(
                    curr_xr_poses["left"],
                    self.snapshot_xr["left"],
                    self.snapshot_robot["left"],
                )
            if "right" in required_keys:
                target_R = self.compute_teleop_transform(
                    curr_xr_poses["right"],
                    self.snapshot_xr["right"],
                    self.snapshot_robot["right"],
                )
            if "head" in required_keys:
                target_Head = self.compute_teleop_transform(
                    curr_xr_poses["head"],
                    self.snapshot_xr["head"],
                    self.snapshot_robot["head"],
                )

            if self.solver is not None:
                # Solve for new configuration using target poses and current config
                # Cast inputs to JAX arrays and output back to numpy
                new_config_jax = self.solver.solve(
                    target_L,
                    target_R,
                    target_Head,
                    jnp.asarray(q_current),
                )
                new_config = np.array(new_config_jax)

                if self.filter is not None:
                    self.filter.add_data(new_config)
                    if self.filter.data_ready():
                        return self.filter.filtered_data

                logger.debug(f"[IKController] New Config: {new_config}")
                return new_config

            return q_current
        else:
            if self.active:
                # Disengagement transition
                self.active = False
                if self.filter is not None:
                    self.filter.reset()
            return q_current
