import numpy as np
import jax.numpy as jnp
import jaxlie
from abc import ABC
from typing import Dict, List, Any
from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController


def test_ik_api_imports():
    assert BaseRobot is not None
    assert PyrokiSolver is not None
    assert IKController is not None


def test_base_robot_is_abc():
    assert issubclass(BaseRobot, ABC)

    abstract_methods = BaseRobot.__abstractmethods__
    expected_methods = {
        "get_vis_config",
        "joint_var_cls",
        "forward_kinematics",
        "get_default_config",
        "build_costs",
    }
    assert expected_methods.issubset(abstract_methods)


def test_pyroki_solver_instantiation():
    class MockRobot(BaseRobot):
        def get_vis_config(self):
            return None

        @property
        def joint_var_cls(self):
            return None

        def forward_kinematics(self, config: jnp.ndarray) -> Dict[str, jaxlie.SE3]:
            return {}

        def get_default_config(self) -> jnp.ndarray:
            return jnp.zeros(1)

        def build_costs(
            self, target_L: jaxlie.SE3, target_R: jaxlie.SE3, target_Head: jaxlie.SE3
        ) -> List[Any]:
            return []

    robot = MockRobot()
    solver = PyrokiSolver(robot)
    assert solver.robot == robot


def test_ik_controller_instantiation():
    class MockRobot(BaseRobot):
        def get_vis_config(self):
            return None

        @property
        def joint_var_cls(self):
            return None

        def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
            return {}

        def get_default_config(self) -> jnp.ndarray:
            return jnp.zeros(2)

        def build_costs(self, target_L, target_R, target_Head):
            return []

    robot = MockRobot()
    controller = IKController(robot=robot)
    assert controller.robot == robot
    assert controller.solver is None
    assert controller.active is False


def test_ik_controller_with_filter():
    class MockRobot(BaseRobot):
        def get_vis_config(self):
            return None

        @property
        def joint_var_cls(self):
            return None

        def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
            return {}

        def get_default_config(self) -> jnp.ndarray:
            return jnp.zeros(2)

        def build_costs(self, target_L, target_R, target_Head):
            return []

    robot = MockRobot()
    filter_weights = np.array([0.5, 0.5])
    controller = IKController(robot=robot, filter_weights=filter_weights)
    assert controller.filter is not None
