import pytest

try:
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import jaxlie  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip(
        "jaxls, pyroki, jaxlie, or yourdfpy not installed", allow_module_level=True
    )

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
from abc import ABC  # noqa: E402
from typing import Dict, List, Any  # noqa: E402
from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController  # noqa: E402


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
        "actuated_joint_names",
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
            self,
            target_L: jaxlie.SE3 | None,
            target_R: jaxlie.SE3 | None,
            target_Head: jaxlie.SE3 | None,
            q_current: jnp.ndarray | None = None,
        ) -> List[Any]:
            return []

        @property
        def actuated_joint_names(self) -> List[str]:
            return ["joint1"]

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

        def build_costs(self, target_L, target_R, target_Head, q_current=None):
            return []

        @property
        def actuated_joint_names(self) -> List[str]:
            return ["joint1", "joint2"]

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

        def build_costs(self, target_L, target_R, target_Head, q_current=None):
            return []

        @property
        def actuated_joint_names(self) -> List[str]:
            return ["joint1", "joint2"]

    robot = MockRobot()
    filter_weights = np.array([0.5, 0.5])
    controller = IKController(robot=robot, filter_weights=filter_weights)
    assert controller.filter is not None
