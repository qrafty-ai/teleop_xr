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

import jax.numpy as jnp  # noqa: E402
from teleop_xr.ik.robot import BaseRobot  # noqa: E402
from teleop_xr.ik.loader import load_robot_class, RobotLoadError  # noqa: E402
from teleop_xr.config import RobotVisConfig  # noqa: E402


class MockCustomRobot(BaseRobot):
    """A minimal robot class for integration testing."""

    def __init__(self, urdf_string=None, custom_arg="default", **kwargs):
        self.urdf_string = urdf_string
        self.custom_arg = custom_arg

    def get_vis_config(self) -> RobotVisConfig | None:
        return None

    @property
    def actuated_joint_names(self) -> list[str]:
        return ["joint1"]

    @property
    def joint_var_cls(self):
        return None

    def forward_kinematics(self, config: jnp.ndarray):
        return {}

    def get_default_config(self) -> jnp.ndarray:
        return jnp.zeros(1)

    def build_costs(self, target_L, target_R, target_Head, q_current=None):
        return []


def test_load_custom_robot_from_module():
    # Test loading this class from this module
    spec = "tests.test_integration_custom_robot:MockCustomRobot"
    cls = load_robot_class(spec)
    assert cls == MockCustomRobot

    # Test instantiation with urdf_string and custom args
    robot = cls(urdf_string="test_urdf", custom_arg="value")
    assert robot.urdf_string == "test_urdf"
    assert robot.custom_arg == "value"


def test_load_invalid_spec():
    with pytest.raises(RobotLoadError, match="Invalid robot specification"):
        load_robot_class("invalid_spec_no_colon")


def test_load_missing_module():
    with pytest.raises(RobotLoadError, match="Failed to load robot class"):
        load_robot_class("non_existent_module:Robot")


def test_load_missing_class():
    with pytest.raises(RobotLoadError, match="Failed to load robot class"):
        load_robot_class("tests.test_integration_custom_robot:NonExistentRobot")


def test_load_not_a_subclass():
    class NotARobot:
        pass

    # We need to make it accessible in the module for load_robot_class to find it if we test via string
    global NotARobotGlobal
    NotARobotGlobal = NotARobot

    with pytest.raises(RobotLoadError, match="is not a subclass of BaseRobot"):
        load_robot_class("tests.test_integration_custom_robot:NotARobotGlobal")
