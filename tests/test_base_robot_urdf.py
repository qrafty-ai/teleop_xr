import pytest

try:
    import jaxlie
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip("IK dependencies not installed", allow_module_level=True)

import jax.numpy as jnp
from typing import Any
from unittest.mock import MagicMock, patch
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.config import RobotVisConfig


class MockRobot(BaseRobot):
    def __init__(self):
        super().__init__()
        self.urdf_path = "mock.urdf"
        self.mesh_path = None

    @property
    def actuated_joint_names(self) -> list[str]:
        return ["joint1"]

    @property
    def joint_var_cls(self) -> Any:
        return MagicMock()

    def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
        return {"left": jaxlie.SE3.identity()}

    def get_default_config(self) -> jnp.ndarray:
        return jnp.zeros(1)

    def build_costs(self, target_L, target_R, target_Head, q_current=None) -> list[Any]:
        return []

    def _load_default_urdf(self):
        mock_urdf = MagicMock()
        mock_urdf.urdf_string = "<robot name='mock'/>"
        return mock_urdf


def test_base_robot_urdf_attributes():
    robot = MockRobot()
    assert hasattr(robot, "urdf_path")
    assert hasattr(robot, "mesh_path")
    assert robot.urdf_path == "mock.urdf"
    assert robot.mesh_path is None


def test_base_robot_model_scale():
    robot = MockRobot()
    assert robot.model_scale == 1.0


def test_base_robot_get_vis_config():
    robot = MockRobot()
    vis_config = robot.get_vis_config()
    assert isinstance(vis_config, RobotVisConfig)
    assert vis_config.urdf_path == "mock.urdf"
    assert vis_config.mesh_path is None
    assert vis_config.model_scale == 1.0
    # Default orientation is identity, so rpy should be [0,0,0]
    assert vis_config.initial_rotation_euler == [0.0, 0.0, 0.0]


@patch("teleop_xr.ram.from_string")
def test_base_robot_load_urdf(mock_ram_from_string):
    # Mock ram.from_string to return (path, mesh)
    mock_ram_from_string.return_value = ("remote.urdf", "remote_meshes")

    robot = MockRobot()

    # Mock yourdfpy.URDF.load if it's used
    with patch("yourdfpy.URDF.load") as mock_urdf_load:
        mock_urdf_load.return_value = MagicMock()

        # Test loading with urdf_string (override)
        robot._load_urdf(urdf_string="<robot/>")

        mock_ram_from_string.assert_called_once_with("<robot/>")
        assert robot.urdf_path == "remote.urdf"
        assert robot.mesh_path == "remote_meshes"
        mock_urdf_load.assert_called_once_with("remote.urdf")


def test_base_robot_load_default_urdf():
    robot = MockRobot()
    urdf = robot._load_urdf()
    assert urdf.urdf_string == "<robot name='mock'/>"
    assert robot.urdf_path == "mock.urdf"


@patch("teleop_xr.ram.from_string")
def test_base_robot_load_urdf_override(mock_ram_from_string):
    mock_ram_from_string.return_value = ("remote.urdf", "remote_meshes")
    robot = MockRobot()

    with patch("yourdfpy.URDF.load") as mock_urdf_load:
        mock_urdf_load.return_value = MagicMock()
        robot._load_urdf(urdf_string="<robot/>")

        mock_ram_from_string.assert_called_once_with("<robot/>")
        assert robot.urdf_path == "remote.urdf"
        assert robot.mesh_path == "remote_meshes"
        mock_urdf_load.assert_called_once_with("remote.urdf")
