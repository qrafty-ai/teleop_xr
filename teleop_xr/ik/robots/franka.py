import io
import os
from typing import Any, override

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost, RobotDescription
from teleop_xr.config import RobotVisConfig
from teleop_xr import ram


class FrankaRobot(BaseRobot):
    """
    Franka Emika Panda robot implementation for IK.
    Single-arm robot mapped to the 'right' controller.
    """

    def __init__(self) -> None:
        self._description_override: RobotDescription | None = None

        # Resolve default URDF path via RAM
        repo_url = "https://github.com/frankarobotics/franka_ros.git"
        xacro_path = "franka_description/robots/panda/panda.urdf.xacro"
        xacro_args = {"hand": "true"}

        self._default_urdf_path = str(
            ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args=xacro_args,
                resolve_packages=True,
            )
        )

        repo_path = ram.get_repo(repo_url)
        self._default_mesh_path: str | None = str(repo_path)

        if not os.path.exists(self._default_urdf_path):
            raise FileNotFoundError(
                f"Franka URDF not found at {self._default_urdf_path}"
            )

        # Robot-specific constants (set before _init_from_description)
        self.ee_link_name = "panda_hand"

        self._init_from_description(self.description)

    # ------------------------------------------------------------------
    # Description management
    # ------------------------------------------------------------------

    @property
    @override
    def description(self) -> RobotDescription:
        if self._description_override is not None:
            return self._description_override
        return RobotDescription(content=self._default_urdf_path, kind="path")

    @override
    def _init_from_description(self, description: RobotDescription) -> None:
        if description.kind == "path":
            self.urdf_path = description.content
            self.mesh_path = (
                os.path.dirname(description.content)
                if description.content
                else self._default_mesh_path
            )
            urdf = yourdfpy.URDF.load(description.content)
        else:
            self.urdf_path = ""
            self.mesh_path = self._default_mesh_path
            urdf = yourdfpy.URDF.load(io.StringIO(description.content))

        self.robot = pk.Robot.from_urdf(urdf)

        # End effector link index
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
