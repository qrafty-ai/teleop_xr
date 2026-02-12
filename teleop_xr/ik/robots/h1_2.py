# pyright: reportCallIssue=false
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
from teleop_xr.config import RobotVisConfig
from teleop_xr import ram


class UnitreeH1Robot(BaseRobot):
    """
    Unitree H1_2 robot implementation for IK.
    """

    def __init__(self) -> None:
        # Load URDF from external repository via RAM
        self.urdf_path = str(
            ram.get_resource(
                repo_url="https://github.com/unitreerobotics/xr_teleoperate.git",
                path_inside_repo="assets/h1_2/h1_2.urdf",
            )
        )
        self.mesh_path = os.path.dirname(self.urdf_path)
        urdf = yourdfpy.URDF.load(self.urdf_path)

        # Identify leg joints names to freeze
        self.leg_joint_names = [
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]

        for joint_name in self.leg_joint_names:
            if joint_name in urdf.joint_map:
                urdf.joint_map[joint_name].type = "fixed"

        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # End effector and torso link indices
        # We use hand base links as end effectors (L_ee, R_ee frames)
        self.L_ee = "L_hand_base_link"
        self.R_ee = "R_hand_base_link"
        self.L_ee_link_idx = self.robot.links.names.index(self.L_ee)
        self.R_ee_link_idx = self.robot.links.names.index(self.R_ee)
        self.torso_link_idx = self.robot.links.names.index("torso_link")

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.identity()

    @override
    def get_vis_config(self) -> RobotVisConfig | None:
        if not self.urdf_path:
            return None
        return RobotVisConfig(
            urdf_path=self.urdf_path,
            mesh_path=self.mesh_path,
            model_scale=0.5,
            initial_rotation_euler=[
                float(x) for x in self.orientation.as_rpy_radians()
            ],
        )

    @property
    def joint_var_cls(self) -> Any:
        """
        The jaxls.Var class used for joint configurations.
        """
        return self.robot.joint_var_cls

    @property
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @property
    def default_speed_ratio(self) -> float:
        # Unitree H1 often benefits from slightly amplified motion mapping
        return 1.2

    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        """
        Compute the forward kinematics for the given configuration.
        """
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
            "head": jaxlie.SE3(fk[self.torso_link_idx]),
        }

    def get_default_config(self) -> jax.Array:
        return jnp.zeros_like(self.robot.joints.lower_limits)

    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        """
        Build a list of Pyroki cost objects.
        """
        costs = []
        JointVar = self.robot.joint_var_cls

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=5.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.01,
            )
        )

        # 1. Bimanual costs (L/R EE frames: L_ee, R_ee)
        # Using analytic jacobian for efficiency
        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        costs.append(
            pk.costs.limit_cost(  # pyright: ignore[reportCallIssue]
                self.robot, JointVar(0), weight=100.0
            )
        )

        if target_Head is not None:
            costs.append(
                pk.costs.pose_cost(  # pyright: ignore[reportCallIssue]
                    robot=self.robot,
                    joint_var=JointVar(0),
                    target_pose=target_Head,
                    target_link_index=jnp.array(self.torso_link_idx, dtype=jnp.int32),
                    pos_weight=0.0,
                    ori_weight=jnp.array([0.0, 0.0, 20.0]),
                )
            )

        return costs
