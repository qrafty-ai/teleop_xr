# pyright: reportCallIssue=false
import os
import io
from pathlib import Path
from typing import Any, override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr.config import RobotVisConfig
from teleop_xr import ram


class TeaArmRobot(BaseRobot):
    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        self.urdf_path: str
        self.mesh_path: str | None
        if urdf_string:
            urdf = yourdfpy.URDF.load(io.StringIO(urdf_string))
            self.urdf_path = ""
            self.mesh_path = None
        else:
            repo_root = Path("/home/cc/codes/tea/ros2_wksp/src/tea-ros2")
            path_inside_repo = "tea_description/urdf/teaarm.urdf.xacro"
            xacro_args = {"with_obstacles": "false", "visual_mesh_ext": "glb"}

            self.urdf_path = str(
                ram.get_resource(
                    repo_root=repo_root,
                    path_inside_repo=path_inside_repo,
                    xacro_args=xacro_args,
                    resolve_packages=True,
                )
            )

            self.mesh_path = str(repo_root)

            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(f"TeaArm URDF not found at {self.urdf_path}")

            urdf = yourdfpy.URDF.load(self.urdf_path)

        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)

        self.L_ee: str = "frame_left_arm_ee"
        self.R_ee: str = "frame_right_arm_ee"

        if self.L_ee in self.robot.links.names:
            self.L_ee_link_idx: int = self.robot.links.names.index(self.L_ee)
        else:
            raise ValueError(f"Link {self.L_ee} not found in URDF")

        if self.R_ee in self.robot.links.names:
            self.R_ee_link_idx: int = self.robot.links.names.index(self.R_ee)
        else:
            raise ValueError(f"Link {self.R_ee} not found in URDF")

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.from_rpy_radians(0.0, 0.0, -1.57079632679)

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right"}

    @override
    def get_vis_config(self) -> RobotVisConfig | None:
        if not self.urdf_path:
            return None
        return RobotVisConfig(
            urdf_path=self.urdf_path,
            mesh_path=self.mesh_path,
            model_scale=1.0,
            initial_rotation_euler=[
                float(x) for x in self.orientation.as_rpy_radians()
            ],
        )

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
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        return jnp.zeros(len(self.actuated_joint_names))

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

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=1.0,
                )
            )

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

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=100.0))

        return costs
