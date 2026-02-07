import pytest
import jax.numpy as jnp
from typing import cast
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.loader import load_robot_class, RobotLoadError
from teleop_xr.config import RobotVisConfig


class MockCustomRobot(BaseRobot):
    """A minimal robot class for integration testing."""

    def __init__(self, urdf_string=None, custom_arg="default", **kwargs):
        super().__init__()
        self.urdf_string = urdf_string
        self.custom_arg = custom_arg

    @property
    def robot_description(self) -> str:
        return self.urdf_string or "<robot name='custom'/>"

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
        del q_current
        return []


def test_load_custom_robot_from_module():
    # Test loading this class from this module
    spec = "tests.test_integration_custom_robot:MockCustomRobot"
    cls = load_robot_class(spec)
    assert cls == MockCustomRobot

    # Test instantiation with urdf_string and custom args
    custom_cls = cast(type[MockCustomRobot], cls)
    robot = custom_cls(urdf_string="test_urdf", custom_arg="value")
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
