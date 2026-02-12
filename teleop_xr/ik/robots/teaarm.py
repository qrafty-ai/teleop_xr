# pyright: reportCallIssue=false
import json
import os
import sys
from pathlib import Path
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


class TeaArmRobot(BaseRobot):
    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        self.urdf_path: str
        self.mesh_path: str | None
        if urdf_string:
            import io

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

        collision_data = self._load_collision_data()
        if collision_data is not None:
            sphere_decomposition, ignore_pairs = collision_data
            self.robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
                sphere_decomposition,
                urdf,
                user_ignore_pairs=ignore_pairs,
                ignore_immediate_adjacents=True,
            )
        else:
            self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

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

        self.waist_link_idx: int = self.robot.links.names.index("waist_link")

    @staticmethod
    def _load_collision_data() -> (
        tuple[dict[str, Any], tuple[tuple[str, str], ...]] | None
    ):
        """Load collision data (spheres + ignore pairs) for TeaArm from assets.

        Tries ``collision.json`` first (new format with ignore pairs),
        falls back to ``sphere.json`` (legacy sphere-only format).

        Returns:
            Tuple of (sphere_decomposition, ignore_pairs) or None.
        """
        asset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "assets",
            "teaarm",
        )

        collision_path = os.path.join(asset_dir, "collision.json")
        sphere_path = os.path.join(asset_dir, "sphere.json")

        try:
            if os.path.exists(collision_path):
                with open(collision_path, "r") as f:
                    data = json.load(f)
                spheres = data["spheres"]
                ignore_pairs = tuple(
                    tuple(pair) for pair in data.get("collision_ignore_pairs", [])
                )
                return spheres, ignore_pairs

            if os.path.exists(sphere_path):
                with open(sphere_path, "r") as f:
                    data = json.load(f)
                return data, ()

        except (json.JSONDecodeError, IOError, KeyError):
            pass

        return None

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.from_rpy_radians(0.0, 0.0, -1.57079632679)

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right", "head"}

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
            "head": jaxlie.SE3(fk[self.waist_link_idx]),
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

        # larger joints are more costly to move
        energy_by_joint = [
            1.0,  # waist_yaw
            10.0,  # waist_pitch
            5.0,  # left_j1
            5.0,  # right_j1
            4.0,  # left_j2
            4.0,  # right_j2
            3.0,  # left_j3
            3.0,  # right_j3
            2.0,  # left_j4
            2.0,  # right_j4
            1.0,  # left_j5
            1.0,  # right_j5
            0.5,  # left_j6
            0.5,  # right_j6
            0.1,  # left_j7
            0.1,  # right_j7
        ]

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=jnp.array(energy_by_joint),
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

        if target_Head is not None:
            costs.append(
                pk.costs.pose_cost(
                    robot=self.robot,
                    joint_var=JointVar(0),
                    target_pose=target_Head,
                    target_link_index=jnp.array(self.waist_link_idx, dtype=jnp.int32),
                    pos_weight=0.0,
                    ori_weight=jnp.array([0.0, 0.0, 20.0]),
                )
            )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=100.0))

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
