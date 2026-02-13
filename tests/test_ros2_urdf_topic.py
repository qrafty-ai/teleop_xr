import pytest

try:
    import jaxlie  # noqa: F401
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip("IK dependencies not installed", allow_module_level=True)

import sys
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

# Mock ROS2 modules before importing teleop_xr.ros2.__main__
mock_rclpy = MagicMock()


class MockNode:
    def __init__(self, *args, **kwargs):
        self._parameters = {}
        self.get_logger = MagicMock()
        self.get_clock = MagicMock()
        self.declare_parameter = MagicMock(
            side_effect=lambda name, value, descriptor=None: self._parameters.update(
                {name: value}
            )
        )

    def create_publisher(self, *args, **kwargs):
        return MagicMock()

    def create_subscription(self, *args, **kwargs):
        return MagicMock()

    def get_parameter(self, name):
        value = self._parameters.get(name)
        param = MagicMock()
        param.value = value
        param.get_parameter_value.return_value = MagicMock()
        if isinstance(value, bool):
            param.get_parameter_value.return_value.bool_value = value
        elif isinstance(value, int):
            param.get_parameter_value.return_value.integer_value = value
        elif isinstance(value, float):
            param.get_parameter_value.return_value.double_value = value
        elif isinstance(value, str):
            param.get_parameter_value.return_value.string_value = value
        return param

    def destroy_node(self):
        pass


mock_rclpy.node.Node = MockNode
sys.modules["rclpy"] = mock_rclpy
sys.modules["rclpy.node"] = mock_rclpy.node
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["sensor_msgs.msg"] = MagicMock()
sys.modules["trajectory_msgs.msg"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["tf2_ros"] = MagicMock()
sys.modules["builtin_interfaces.msg"] = MagicMock()
sys.modules["cv_bridge"] = MagicMock()

import importlib  # noqa: E402
import teleop_xr.ros2.__main__  # noqa: E402

importlib.reload(teleop_xr.ros2.__main__)
from teleop_xr.ros2.__main__ import main, TeleopNode, Ros2CLI  # noqa: E402

from teleop_xr.ik.robot import BaseRobot  # noqa: E402


class MockRobot(BaseRobot):
    def __init__(self, urdf_string=None, **kwargs):
        super().__init__()
        self.urdf_string_received = urdf_string
        # Minimal setup to make get_vis_config work
        if urdf_string:
            self.urdf_path = "/tmp/mock_overridden.urdf"
        else:
            self.urdf_path = "/path/to/default.urdf"
        self._actuated_joint_names = ["joint1"]

    def _load_default_urdf(self):
        return MagicMock()

    def get_default_config(self):
        return jnp.array([0.0])

    @property
    def actuated_joint_names(self):
        return self._actuated_joint_names

    @property
    def joint_var_cls(self):
        return MagicMock()

    def forward_kinematics(self, config):
        return {}

    def build_costs(self, *args, **kwargs):
        return []


@patch("teleop_xr.ros2.__main__.load_robot_class")
@patch("teleop_xr.ros2.__main__.get_urdf_from_topic")
@patch("teleop_xr.ros2.__main__.Teleop")
@patch("teleop_xr.ros2.__main__.IKWorker")
@patch("rclpy.init")
@patch("rclpy.spin")
def test_urdf_topic_integration(
    mock_spin, mock_init, mock_ik_worker, mock_teleop, mock_get_urdf, mock_load_robot
):
    # Setup mocks
    mock_load_robot.return_value = MockRobot
    mock_get_urdf.return_value = "<robot name='overridden'/>"

    # Create a CLI object
    cli = Ros2CLI(
        mode="ik",
        robot_class="MockRobot",
        urdf_topic="/robot_description",
        no_urdf_topic=False,
    )

    node_instance = TeleopNode(cli)

    with patch("teleop_xr.ros2.__main__.tyro.cli", return_value=cli):
        with patch("teleop_xr.ros2.__main__.TeleopNode", return_value=node_instance):
            # Run main
            # We need to catch SystemExit or prevent infinite loops if teleop.run() blocks
            mock_teleop_instance = mock_teleop.return_value
            # Prevent teleop.run() from blocking
            mock_teleop_instance.run.return_value = None

            main()

            # Verify get_urdf_from_topic was called
            mock_get_urdf.assert_called_once_with(
                node_instance, "/robot_description", 5.0
            )

            # Verify Teleop call
            args, kwargs = mock_teleop.call_args
            settings = kwargs["settings"]
            assert settings.robot_vis is not None
            # In our MockRobot, we set urdf_path to /tmp/mock_overridden.urdf if urdf_string is provided
            assert settings.robot_vis.urdf_path == "/tmp/mock_overridden.urdf"


@patch("teleop_xr.ros2.__main__.load_robot_class")
@patch("teleop_xr.ros2.__main__.get_urdf_from_topic")
@patch("teleop_xr.ros2.__main__.Teleop")
@patch("teleop_xr.ros2.__main__.IKWorker")
@patch("rclpy.init")
@patch("rclpy.spin")
def test_no_urdf_topic_integration(
    mock_spin, mock_init, mock_ik_worker, mock_teleop, mock_get_urdf, mock_load_robot
):
    # Setup mocks
    mock_load_robot.return_value = MockRobot

    cli = Ros2CLI(
        mode="ik",
        robot_class="MockRobot",
        no_urdf_topic=True,  # Disable URDF topic
    )
    node_instance = TeleopNode(cli)

    with patch("teleop_xr.ros2.__main__.tyro.cli", return_value=cli):
        with patch("teleop_xr.ros2.__main__.TeleopNode", return_value=node_instance):
            mock_teleop_instance = mock_teleop.return_value
            mock_teleop_instance.run.return_value = None

            main()

            # Verify get_urdf_from_topic was NOT called
            mock_get_urdf.assert_not_called()

            # Verify Teleop call
            args, kwargs = mock_teleop.call_args
            settings = kwargs["settings"]
            assert settings.robot_vis is not None
            # In our MockRobot, we set urdf_path to /path/to/default.urdf if NO urdf_string is provided
            assert settings.robot_vis.urdf_path == "/path/to/default.urdf"
