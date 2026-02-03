import io
import os
from typing import Any

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr.config import RobotVisConfig


class FrankaRobot(BaseRobot):
    """
    Franka Emika Panda robot implementation for IK.
    Single-arm robot mapped to the 'right' controller.
    """

    def __init__(self, urdf_string: str | None = None) -> None:
        if urdf_string:
            urdf = yourdfpy.URDF.load(io.StringIO(urdf_string))
            self.urdf_path = ""
            self.mesh_path = ""
        else:
            # Load URDF from assets
            self.urdf_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "assets",
                    "franka",
                    "panda.urdf",
                )
            )
            if not os.path.exists(self.urdf_path):
                # Fallback
                self.urdf_path = os.path.join(
                    "teleop_xr", "assets", "franka", "panda.urdf"
                )

            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(f"Franka URDF not found at {self.urdf_path}")

            self.mesh_path = os.path.dirname(self.urdf_path)
            urdf = yourdfpy.URDF.load(self.urdf_path)

        self.robot = pk.Robot.from_urdf(urdf)

        # End effector link index
        self.ee_link_name = "panda_hand"
        if self.ee_link_name in self.robot.links.names:
            self.ee_link_idx = self.robot.links.names.index(self.ee_link_name)
        else:
            # Fallback if hand not present (e.g. simplified URDF), use last link
            self.ee_link_idx = len(self.robot.links.names) - 1

    @property
    def supported_frames(self) -> set[str]:
        return {"right"}

    def get_vis_config(self) -> RobotVisConfig | None:
        if not self.urdf_path:
            return None
        return RobotVisConfig(
            urdf_path=self.urdf_path,
            mesh_path=self.mesh_path,
            model_scale=1.0,
            initial_rotation_euler=[0.0, 0.0, 0.0],
        )

    @property
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "right": jaxlie.SE3(fk[self.ee_link_idx]),
        }

    def get_default_config(self) -> jax.Array:
        # Default home pose
        q = jnp.zeros_like(self.robot.joints.lower_limits)
        # Set some reasonable angles if possible (e.g. elbow bent)
        # but zeros is fine for now
        return q

    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls

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
