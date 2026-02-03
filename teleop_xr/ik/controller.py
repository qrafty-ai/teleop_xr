import jax.numpy as jnp
import jaxlie
import numpy as np
from teleop_xr.utils.filter import WeightedMovingFilter
from teleop_xr.messages import XRState, XRDeviceRole, XRHandedness, XRPose
from typing import Optional, Dict, Any


class IKController:
    def __init__(
        self,
        robot: Any,
        solver: Any = None,
        filter_weights: Optional[np.ndarray] = None,
    ):
        self.robot = robot
        self.solver = solver
        self.active = False

        # Snapshots
        self.snapshot_xr: Dict[str, jaxlie.SE3] = {}
        self.snapshot_robot: Dict[str, jaxlie.SE3] = {}

        # Filter for joint configuration
        self.filter: Optional[WeightedMovingFilter] = None
        if filter_weights is not None:
            # We'll initialize the filter when we know the data size (from default config)
            default_config = self.robot.get_default_config()
            self.filter = WeightedMovingFilter(
                filter_weights, data_size=len(default_config)
            )

    def xr_pose_to_se3(self, pose: XRPose) -> jaxlie.SE3:
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
        R_xr_to_robot = jaxlie.SO3.from_matrix(
            jnp.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )

        t_delta_xr = t_ctrl_curr.translation() - t_ctrl_init.translation()
        t_delta_robot = R_xr_to_robot @ t_delta_xr

        print(f"[IKController] t_delta_xr: {t_delta_xr}")
        print(f"[IKController] t_delta_robot: {t_delta_robot}")

        q_delta_xr = t_ctrl_curr.rotation() @ t_ctrl_init.rotation().inverse()
        q_delta_robot = R_xr_to_robot @ q_delta_xr @ R_xr_to_robot.inverse()

        t_new = t_ee_init.translation() + t_delta_robot
        q_new = q_delta_robot @ t_ee_init.rotation()

        return jaxlie.SE3.from_rotation_and_translation(q_new, t_new)

    def _get_device_poses(self, state: XRState) -> Dict[str, jaxlie.SE3]:
        poses = {}
        for device in state.devices:
            if device.role == XRDeviceRole.CONTROLLER:
                if device.handedness == XRHandedness.LEFT and device.gripPose:
                    poses["left"] = self.xr_pose_to_se3(device.gripPose)
                elif device.handedness == XRHandedness.RIGHT and device.gripPose:
                    poses["right"] = self.xr_pose_to_se3(device.gripPose)
            elif device.role == XRDeviceRole.HEAD and device.pose:
                poses["head"] = self.xr_pose_to_se3(device.pose)
        return poses

    def _check_deadman(self, state: XRState) -> bool:
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
        print("[IKController] Reset triggered")

    def step(self, state: XRState, current_config: np.ndarray) -> np.ndarray:
        is_deadman_active = self._check_deadman(state)
        curr_xr_poses = self._get_device_poses(state)

        # Check if we have all necessary poses
        required_keys = ["left", "right", "head"]
        has_all_poses = all(k in curr_xr_poses for k in required_keys)

        if is_deadman_active and has_all_poses:
            if not self.active:
                # Engagement transition: take snapshots
                self.active = True
                self.snapshot_xr = curr_xr_poses

                # Get initial robot FK poses
                fk_poses = self.robot.forward_kinematics(current_config)
                # We expect fk_poses to be a dict with "left", "right", "head" keys
                self.snapshot_robot = {
                    "left": fk_poses["left"],
                    "right": fk_poses["right"],
                    "head": fk_poses["head"],
                }

                print(f"[IKController] Initial Robot FK: {self.snapshot_robot}")
                return current_config

            # Active control
            target_L = self.compute_teleop_transform(
                curr_xr_poses["left"],
                self.snapshot_xr["left"],
                self.snapshot_robot["left"],
            )
            target_R = self.compute_teleop_transform(
                curr_xr_poses["right"],
                self.snapshot_xr["right"],
                self.snapshot_robot["right"],
            )
            target_Head = self.compute_teleop_transform(
                curr_xr_poses["head"],
                self.snapshot_xr["head"],
                self.snapshot_robot["head"],
            )

            print(f"[IKController] Target L: {target_L.translation()}")
            print(f"[IKController] Target R: {target_R.translation()}")

            if self.solver is not None:
                # Solve for new configuration using target poses and current config
                new_config = self.solver.solve(
                    target_L, target_R, target_Head, current_config
                )

                if self.filter is not None:
                    self.filter.add_data(new_config)
                    if self.filter.data_ready():
                        return self.filter.filtered_data

                return new_config

            return current_config
        else:
            if self.active:
                # Disengagement transition
                self.active = False
                if self.filter is not None:
                    self.filter.reset()
            return current_config
