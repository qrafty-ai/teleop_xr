import os
import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr import ram


class SO101Robot(BaseRobot):
    """
    SO-ARM100 (SO101) robot implementation for IK.
    Single-arm robot with 5 DOF (excluding gripper) mapped to the 'right' controller.

    Note: SO101 has only 5 rotational DOF in the arm, so the IK solution is
    relaxed to prioritize translation over full 6D pose satisfaction.
    """

    # URDF repository and path
    URDF_REPO_URL = "https://github.com/TheRobotStudio/SO-ARM100.git"
    URDF_PATH = "Simulation/SO101/so101_new_calib.urdf"

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        urdf = self._load_urdf(urdf_string)

        # Fix gripper joint to make it passive (not actively controlled)
        # The gripper joint is "gripper" - we fix it to a neutral position
        if "gripper" in urdf.joint_map:
            urdf.joint_map["gripper"].type = "fixed"
            urdf.joint_map["gripper"].mimic = None
        urdf._update_actuated_joints()

        self.robot = pk.Robot.from_urdf(urdf)

        # End effector link index - use gripper_link as the end-effector
        self.ee_link_name = "gripper_link"
        if self.ee_link_name in self.robot.links.names:
            self.ee_link_idx = self.robot.links.names.index(self.ee_link_name)
        else:
            # Fallback if not found, use last link
            self.ee_link_idx = len(self.robot.links.names) - 1

    def _load_default_urdf(self) -> yourdfpy.URDF:
        # Load URDF via RAM from GitHub repository
        self.urdf_path = str(
            ram.get_resource(
                repo_url=self.URDF_REPO_URL,
                path_inside_repo=self.URDF_PATH,
                resolve_packages=True,
            )
        )

        # Set mesh path to the repo root in RAM cache
        repo_path = ram.get_repo(self.URDF_REPO_URL)
        self.mesh_path = str(repo_path)

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"SO101 URDF not found at {self.urdf_path}")

        return yourdfpy.URDF.load(self.urdf_path)

    @property
    @override
    def model_scale(self) -> float:
        return 1.0

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        # SO101 base orientation - likely needs no rotation adjustment
        # Based on URDF, the base is aligned with ROS conventions
        return jaxlie.SO3.identity()

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"right"}

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "right": jaxlie.SE3(fk[self.ee_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        # Default home pose for SO101
        # 5 arm joints + 1 gripper joint = 6 total
        # Home position: all zeros (neutral pose)
        q = jnp.zeros(6)

        # If there are more joints, pad with zeros
        n_actuated = len(self.robot.joints.actuated_names)
        if len(q) < n_actuated:
            q = jnp.pad(q, (0, n_actuated - len(q)))
        elif len(q) > n_actuated:
            q = q[:n_actuated]

        return q

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls

        # Rest cost for smooth motion (regularization)
        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=1.0,
                )
            )

        # Manipulability cost to avoid singularities
        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.ee_link_idx], dtype=jnp.int32),
                weight=0.001,
            )
        )

        # Pose cost for right arm (SO101 is a 5-DOF arm)
        # NOTE: With only 5 DOF, we cannot achieve arbitrary 6D poses.
        # We prioritize translation (position) over rotation (orientation).
        # The ori_weight is significantly reduced compared to 7-DOF robots
        # like Franka Panda to allow the solver to focus on translation.
        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,  # High priority for translation
                    ori_weight=1.0,  # Low priority for rotation (5-DOF limitation)
                )
            )

        # Joint limit cost to keep within safe ranges
        costs.append(
            pk.costs.limit_cost(  # pyright: ignore[reportCallIssue]
                self.robot, JointVar(0), weight=100.0
            )
        )

        return costs
