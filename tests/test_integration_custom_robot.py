import pytest
import jax.numpy as jnp
from teleop_xr.ik.robot import BaseRobot, RobotDescription
from teleop_xr.ik.loader import load_robot_class, RobotLoadError
from teleop_xr.config import RobotVisConfig


class MockCustomRobot(BaseRobot):
    def __init__(self, custom_arg="default", **kwargs):
        self._description_override: RobotDescription | None = None
        self.custom_arg = custom_arg
        self._init_from_description(self.description)

    @property
    def description(self) -> RobotDescription:
        if self._description_override is not None:
            return self._description_override
        return RobotDescription(content="<robot name='mock'/>", kind="urdf_string")

    def _init_from_description(self, description: RobotDescription) -> None:
        self._current_description = description

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
    spec = "tests.test_integration_custom_robot:MockCustomRobot"
    cls = load_robot_class(spec)
    assert cls == MockCustomRobot

    robot = MockCustomRobot(custom_arg="value")
    assert robot.custom_arg == "value"
    assert robot.description.kind == "urdf_string"

    robot.set_description("<robot name='overridden'/>")
    assert robot.description.kind == "urdf_string"
    assert "overridden" in robot.description.content


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
