# pyright: reportCallIssue=false
import io
import os
from typing import Any, override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr.config import RobotVisConfig
from teleop_xr import ram


class OpenArmRobot(BaseRobot):
    """
    OpenArm bimanual robot implementation for IK.
    Uses the openarm_description package with bimanual=true.
    """

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        self.mesh_path: str | None = None

        if urdf_string:
            urdf = yourdfpy.URDF.load(io.StringIO(urdf_string))
            self.urdf_path = ""
        else:
            repo_url = "https://github.com/enactic/openarm_description.git"
            xacro_path = "urdf/robot/v10.urdf.xacro"
            xacro_args = {
                "bimanual": "true",
                "hand": "false",
                "ros2_control": "false",
            }

            self.urdf_path = str(
                ram.get_resource(
                    repo_url=repo_url,
                    path_inside_repo=xacro_path,
                    xacro_args=xacro_args,
                    resolve_packages=True,
                )
            )

            repo_path = ram.get_repo(repo_url)
            self.mesh_path = str(repo_path)

            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(
                    f"OpenArm URDF not found at {self.urdf_path}"
                )

            urdf = yourdfpy.URDF.load(self.urdf_path)

        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        # End effector links for bimanual setup
        self.L_ee: str = "openarm_left_link7"
        self.R_ee: str = "openarm_right_link7"

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
                    weight=5.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array(
                    [self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32
                ),
                weight=0.01,
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

        costs.append(
            pk.costs.limit_cost(self.robot, JointVar(0), weight=100.0)
        )

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs
