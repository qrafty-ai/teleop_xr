# pyright: reportCallIssue=false
import os
from typing import Any

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

        # End effector and torso link indices
        # We use hand base links as end effectors (L_ee, R_ee frames)
        self.L_ee = "L_hand_base_link"
        self.R_ee = "R_hand_base_link"
        self.L_ee_link_idx = self.robot.links.names.index(self.L_ee)
        self.R_ee_link_idx = self.robot.links.names.index(self.R_ee)
        self.torso_link_idx = self.robot.links.names.index("torso_link")

    def get_vis_config(self) -> RobotVisConfig:
        return RobotVisConfig(
            urdf_path=self.urdf_path,
            mesh_path=self.mesh_path,
            model_scale=0.5,
            initial_rotation_euler=[0.0, 0.0, 1.57079632679],  # Math.PI / 2
        )

    @property
    def joint_var_cls(self) -> Any:
        """
        The jaxls.Var class used for joint configurations.
        """
        return self.robot.joint_var_cls

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
        self, target_L: jaxlie.SE3, target_R: jaxlie.SE3, target_Head: jaxlie.SE3
    ) -> list[Cost]:
        """
        Build a list of Pyroki cost objects.
        """
        costs = []
        JointVar = self.robot.joint_var_cls

        # 1. Bimanual costs (L/R EE frames: L_ee, R_ee)
        # Using analytic jacobian for efficiency
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
