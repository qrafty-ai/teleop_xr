# pyright: reportCallIssue=false
import json
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

from teleop_xr import ram
from teleop_xr.ik.robot import BaseRobot, Cost


class TeaArmRobot(BaseRobot):
    URDF_REPO_URL = "https://github.com/qrafty-ai/hardware_designs.git"
    URDF_PATH = "teaarm_description/urdf/teaarm_asm.urdf.xacro"

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        urdf = self._load_urdf(urdf_string)
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

        self.L_ee = "frame_left_arm_ee"
        self.R_ee = "frame_right_arm_ee"

        if self.L_ee in self.robot.links.names:
            self.L_ee_link_idx = self.robot.links.names.index(self.L_ee)
        else:
            raise ValueError(f"Link {self.L_ee} not found in URDF")

        if self.R_ee in self.robot.links.names:
            self.R_ee_link_idx = self.robot.links.names.index(self.R_ee)
        else:
            raise ValueError(f"Link {self.R_ee} not found in URDF")

        self.waist_link_idx = self.robot.links.names.index("waist_link")

    @override
    def _load_default_urdf(self) -> yourdfpy.URDF:
        repo_root = ram.get_repo(repo_url=self.URDF_REPO_URL)
        self.urdf_path = str(
            ram.get_resource(
                repo_url=self.URDF_REPO_URL,
                path_inside_repo=self.URDF_PATH,
                xacro_args={"visual_mesh_ext": "glb"},
                resolve_packages=True,
            )
        )
        self.mesh_path = str(repo_root / "teaarm_description")
        return yourdfpy.URDF.load(self.urdf_path)

    @staticmethod
    def _load_collision_data() -> (
        tuple[dict[str, Any], tuple[tuple[str, str], ...]] | None
    ):
        asset_dir = Path(__file__).resolve().parent / "assets" / "teaarm"

        collision_path = asset_dir / "collision.json"
        sphere_path = asset_dir / "sphere.json"

        try:
            if collision_path.exists():
                with collision_path.open() as f:
                    data = json.load(f)
                spheres = data["spheres"]
                ignore_pairs = tuple(
                    tuple(pair) for pair in data.get("collision_ignore_pairs", [])
                )
                return spheres, ignore_pairs

            if sphere_path.exists():
                with sphere_path.open() as f:
                    data = json.load(f)
                return data, ()
        except (OSError, json.JSONDecodeError, KeyError):
            return None

        return None

    @property
    @override
    def orientation(self) -> jaxlie.SO3:
        return jaxlie.SO3.from_rpy_radians(0.0, 0.0, -1.57079632679)

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right"}

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
        initial_positions = {
            "left_j1": 1.15191731,
            "left_j2": -0.33161256,
            "left_j3": -0.27925268,
            "left_j4": 1.48352986,
            "left_j5": -0.31415927,
            "left_j6": -0.29670597,
            "left_j7": -1.11701072,
            "right_j1": -1.15191731,
            "right_j2": 0.33161256,
            "right_j3": 0.27925268,
            "right_j4": 1.48352986,
            "right_j5": 0.31415927,
            "right_j6": 0.29670597,
            "right_j7": 1.11701072,
        }

        return jnp.array(
            [initial_positions.get(name, 0.0) for name in self.actuated_joint_names]
        )

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

        energy_by_joint = [
            5.0,
            7.0,
            5.0,
            5.0,
            4.0,
            4.0,
            3.0,
            3.0,
            0.5,
            0.5,
            0.3,
            0.3,
            0.5,
            0.5,
            0.1,
            0.1,
        ]

        if q_current is not None:
            n_joints = len(self.actuated_joint_names)
            if len(energy_by_joint) < n_joints:
                energy_weights = energy_by_joint + [1.0] * (
                    n_joints - len(energy_by_joint)
                )
            else:
                energy_weights = energy_by_joint[:n_joints]

            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=jnp.array(energy_weights),
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.005,
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

        centering_weight = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
            0.0,
            0.0,
            0.1,
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        n_joints = len(self.actuated_joint_names)
        if len(centering_weight) < n_joints:
            centering_weights = centering_weight + [0.0] * (
                n_joints - len(centering_weight)
            )
        else:
            centering_weights = centering_weight[:n_joints]

        costs.append(
            pk.costs.rest_cost(
                JointVar(0),
                rest_pose=jnp.zeros(n_joints),
                weight=jnp.array(centering_weights),
            )
        )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=50.0))

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
