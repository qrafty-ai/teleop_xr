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


class FrankaRobot(BaseRobot):
    """
    Franka Emika Panda robot implementation for IK.
    Single-arm robot mapped to the 'right' controller.
    """

    def __init__(
        self,
        robot_description_override: str | None = None,
        urdf_string: str | None = None,
    ) -> None:
        if robot_description_override is not None and urdf_string is not None:
            raise ValueError(
                "Provide only one of robot_description_override or urdf_string"
            )

        repo_url = "https://github.com/frankarobotics/franka_ros.git"
        xacro_path = "franka_description/robots/panda/panda.urdf.xacro"
        xacro_args = {"hand": "true"}
        self._repo_url = repo_url
        self._xacro_path = xacro_path
        self._xacro_args = xacro_args

        self._default_robot_description: str | None = None
        self._default_mesh_path: str | None = None
        self.urdf_path = ""
        self.mesh_path: str | None = None

        super().__init__(robot_description_override or urdf_string)
        self.reinitialize_from_description()

    @property
    @override
    def robot_description(self) -> str:
        if self._default_robot_description is None:
            self._default_robot_description = str(
                ram.get_resource(
                    repo_url=self._repo_url,
                    path_inside_repo=self._xacro_path,
                    xacro_args=self._xacro_args,
                    resolve_packages=True,
                )
            )
            self._default_mesh_path = str(ram.get_repo(self._repo_url))
        return self._default_robot_description

    @override
    def _initialize_from_description(self, robot_description: str) -> None:
        if robot_description.lstrip().startswith("<"):
            self.urdf_path = ""
            self.mesh_path = None
            urdf = yourdfpy.URDF.load(io.StringIO(robot_description))
        elif os.path.exists(robot_description):
            self.urdf_path = robot_description
            if (
                self._default_robot_description is not None
                and robot_description == self._default_robot_description
            ):
                self.mesh_path = self._default_mesh_path
            else:
                self.mesh_path = os.path.dirname(robot_description)
            urdf = yourdfpy.URDF.load(robot_description)
        else:
            raise FileNotFoundError(
                f"Franka robot description path not found: {robot_description}"
            )

        self.robot = pk.Robot.from_urdf(urdf)

        # End effector link index
        self.ee_link_name = "panda_hand"
        if self.ee_link_name in self.robot.links.names:
            self.ee_link_idx = self.robot.links.names.index(self.ee_link_name)
        else:
            # Fallback if hand not present (e.g. simplified URDF), use last link
            self.ee_link_idx = len(self.robot.links.names) - 1

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.identity()

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"right"}

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
        # Default home pose (non-singular)
        # Standard Franka Panda home pose: [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]
        # q1..q7 (7 joints)
        q_target = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        q = jnp.array(q_target)

        # If there are more joints (e.g. gripper finger1, finger2), pad with zeros
        # Typically panda has 7 arm joints + 2 finger joints = 9
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

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=1.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.ee_link_idx], dtype=jnp.int32),
                weight=0.001,
            )
        )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        costs.append(
            pk.costs.limit_cost(  # pyright: ignore[reportCallIssue]
                self.robot, JointVar(0), weight=100.0
            )
        )

        return costs
