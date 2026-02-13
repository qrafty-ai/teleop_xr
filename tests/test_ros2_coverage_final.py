import sys
import numpy as np
from unittest.mock import MagicMock, patch, ANY

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
        self.create_subscription = MagicMock()
        self.create_publisher = MagicMock()
        self.destroy_subscription = MagicMock()

    def get_parameter(self, name):
        value = self._parameters.get(name)
        param = MagicMock()
        param.value = value
        return param

    def destroy_node(self):
        pass


class MockMsg:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CompressedImage(MockMsg):
    pass


class Image(MockMsg):
    pass


class Joy(MockMsg):
    pass


class JointState(MockMsg):
    pass


sensor_msgs = MagicMock()
sensor_msgs.msg.CompressedImage = CompressedImage
sensor_msgs.msg.Image = Image
sensor_msgs.msg.Joy = Joy
sensor_msgs.msg.JointState = JointState

mock_rclpy.node.Node = MockNode
sys.modules["rclpy"] = mock_rclpy
sys.modules["rclpy.node"] = mock_rclpy.node
sys.modules["rclpy.qos"] = MagicMock()
sys.modules["geometry_msgs.msg"] = MagicMock()
sys.modules["sensor_msgs.msg"] = sensor_msgs.msg
sys.modules["trajectory_msgs.msg"] = MagicMock()
sys.modules["std_msgs.msg"] = MagicMock()
sys.modules["tf2_ros"] = MagicMock()
sys.modules["builtin_interfaces.msg"] = MagicMock()
sys.modules["cv_bridge"] = MagicMock()

import importlib
import teleop_xr.ros2.__main__ as ros2_main

importlib.reload(ros2_main)
from teleop_xr.ros2.__main__ import (
    RosBridgeHandler,
    pose_dict_to_matrix,
    matrix_to_pose_msg,
    ms_to_time,
    get_urdf_from_topic,
    ROSImageToVideoSource,
    build_joy,
    IKWorker,
    Ros2CLI,
    TeleopNode,
    main,
)
from teleop_xr.messages import XRState


def test_ros_bridge_handler():
    node = MockNode()
    handler = RosBridgeHandler(node)

    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for level in levels:
        message = MagicMock()
        message.record = {"level": MagicMock(), "message": f"test {level}"}
        message.record["level"].name = level
        handler.write(message)

    assert node.get_logger().debug.called
    assert node.get_logger().info.called
    assert node.get_logger().warn.called
    assert node.get_logger().error.called
    assert node.get_logger().fatal.called


def test_pose_helpers():
    assert pose_dict_to_matrix(None) is None
    assert pose_dict_to_matrix({}) is None

    pose = {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    mat = pose_dict_to_matrix(pose)
    assert mat is not None
    assert mat[0, 3] == 1.0

    with patch("teleop_xr.ros2.__main__.Pose") as mock_pose_cls:
        mock_pose = mock_pose_cls.return_value
        msg = matrix_to_pose_msg(mat)
        assert msg == mock_pose
        assert msg.position.x == 1.0


def test_ms_to_time():
    with patch("teleop_xr.ros2.__main__.Time") as mock_time_cls:
        ms_to_time(1500)
        mock_time_cls.assert_called_with(sec=1, nanosec=500000000)


def test_get_urdf_from_topic():
    node = MockNode()

    with patch("teleop_xr.ros2.__main__.rclpy.ok", side_effect=[True, False]):

        def mock_callback(msg_type, topic, callback, qos=None):
            msg = MagicMock()
            msg.data = "<robot/>"
            callback(msg)
            return MagicMock()

        node.create_subscription.side_effect = mock_callback

        urdf = get_urdf_from_topic(node, "/urdf", 1.0)
        assert urdf == "<robot/>"


def test_ros_image_to_video_source():
    node = MockNode()
    source = MagicMock()

    with (
        patch("teleop_xr.ros2.__main__.CompressedImage", CompressedImage),
        patch("teleop_xr.ros2.__main__.Image", Image),
        patch("teleop_xr.ros2.__main__.HAS_CV_BRIDGE", True),
    ):
        handler = ROSImageToVideoSource(node, "/cam/compressed", source)
        msg = CompressedImage()
        handler.callback(msg)
        assert handler.bridge.compressed_imgmsg_to_cv2.called
        assert source.put_frame.called

    source.reset_mock()
    with (
        patch("teleop_xr.ros2.__main__.CompressedImage", CompressedImage),
        patch("teleop_xr.ros2.__main__.Image", Image),
        patch("teleop_xr.ros2.__main__.HAS_CV_BRIDGE", False),
    ):
        handler = ROSImageToVideoSource(node, "/cam/compressed", source)
        msg = CompressedImage()
        handler.callback(msg)
        assert not source.put_frame.called

    source.reset_mock()
    with (
        patch("teleop_xr.ros2.__main__.CompressedImage", CompressedImage),
        patch("teleop_xr.ros2.__main__.Image", Image),
        patch("teleop_xr.ros2.__main__.HAS_CV_BRIDGE", False),
    ):
        handler = ROSImageToVideoSource(node, "/cam", source)

        msg = Image(encoding="rgb8", height=10, width=10, data=bytes(300))
        handler.callback(msg)
        assert source.put_frame.called

        source.reset_mock()
        msg.encoding = "mono8"
        msg.data = bytes(100)
        handler.callback(msg)
        assert source.put_frame.called

        source.reset_mock()
        msg.encoding = "yuv422"
        handler.callback(msg)
        assert not source.put_frame.called


def test_build_joy():
    assert build_joy(None) == (None, None)
    gamepad = {
        "buttons": [{"pressed": True, "touched": True, "value": 1.0}],
        "axes": [0.5],
    }
    (buttons, axes), touched = build_joy(gamepad)
    assert buttons == [1]
    assert axes == [0.5, 1.0]
    assert touched == [1]


@patch("teleop_xr.ros2.__main__.JointTrajectory")
@patch("teleop_xr.ros2.__main__.JointTrajectoryPoint")
@patch("teleop_xr.ros2.__main__.Duration")
def test_ik_worker(mock_duration, mock_point_cls, mock_traj_cls):
    controller = MagicMock()
    robot = MagicMock()
    robot.actuated_joint_names = ["j1"]
    publisher = MagicMock()
    state_container = {"q": np.zeros(1)}
    node = MockNode()

    worker = IKWorker(controller, robot, publisher, state_container, node)

    state = XRState(timestamp_unix_ms=1000, devices=[])
    worker.update_state(state)
    assert worker.latest_xr_state == state
    assert worker.new_state_event.is_set()

    controller.step.return_value = np.array([0.1])
    controller.active = True

    with patch.object(worker.new_state_event, "wait", side_effect=[True, False]):

        def stop_running(*args, **kwargs):
            worker.running = False
            return True

        worker.new_state_event.wait = stop_running

        worker.run()

    assert state_container["q"][0] == 0.1
    assert publisher.publish.called


def test_main_list_robots():
    cli = Ros2CLI(list_robots=True)
    with patch("teleop_xr.ros2.__main__.tyro.cli", return_value=cli):
        with patch(
            "teleop_xr.ros2.__main__.list_available_robots",
            return_value={"franka": "path"},
        ):
            main()


def test_robot_args_dict_error():
    cli = Ros2CLI(robot_args="{invalid}")
    assert cli.robot_args_dict == {}


def test_ik_worker_set_loop():
    worker = IKWorker(MagicMock(), MagicMock(), MagicMock(), {}, MockNode())
    worker.set_teleop_loop("loop")
    assert worker.teleop_loop == "loop"


@patch("teleop_xr.ros2.__main__.load_robot_class")
@patch("teleop_xr.ros2.__main__.get_urdf_from_topic")
@patch("teleop_xr.ros2.__main__.Teleop")
@patch("teleop_xr.ros2.__main__.IKWorker")
@patch("rclpy.init")
@patch("rclpy.spin")
@patch("teleop_xr.ros2.__main__.TransformBroadcaster")
def test_main_ik_mode(
    mock_broadcaster,
    mock_spin,
    mock_init,
    mock_ik_worker,
    mock_teleop,
    mock_get_urdf,
    mock_load_robot,
):
    mock_robot_cls = MagicMock()
    mock_robot_cls.__name__ = "MockRobot"
    mock_robot = mock_robot_cls.return_value
    mock_robot.actuated_joint_names = ["j1"]
    mock_robot.get_default_config.return_value = [0.0]
    mock_robot.get_vis_config.return_value = None
    mock_load_robot.return_value = mock_robot_cls
    mock_get_urdf.return_value = "<robot/>"

    cli = Ros2CLI(mode="ik", robot_class="Franka")
    node = TeleopNode(cli)

    with patch("teleop_xr.ros2.__main__.tyro.cli", return_value=cli):
        with patch("teleop_xr.ros2.__main__.TeleopNode", return_value=node):
            mock_teleop_inst = mock_teleop.return_value
            main()

            mock_robot_cls.assert_called()

            call_args_list = node.create_subscription.call_args_list
            joint_state_cb = None
            for call_args in call_args_list:
                if call_args[0][1] == "/joint_states":
                    joint_state_cb = call_args[0][2]
                    break

            assert joint_state_cb is not None

            worker_inst = mock_ik_worker.return_value
            worker_inst.teleop = mock_teleop_inst
            worker_inst.teleop_loop = MagicMock()

            msg = JointState(name=["j1"], position=[0.5])
            with patch(
                "teleop_xr.ros2.__main__.asyncio.run_coroutine_threadsafe"
            ) as mock_run:
                joint_state_cb(msg)
                assert mock_run.called


@patch("teleop_xr.ros2.__main__.Teleop")
@patch("teleop_xr.ros2.__main__.IKWorker")
@patch("rclpy.init")
@patch("rclpy.spin")
@patch("teleop_xr.ros2.__main__.TransformBroadcaster")
def test_main_teleop_callback_full(
    mock_broadcaster, mock_spin, mock_init, mock_ik_worker, mock_teleop
):
    cli = Ros2CLI(mode="teleop", publish_hand_tf=True)
    node = TeleopNode(cli)

    with patch("teleop_xr.ros2.__main__.tyro.cli", return_value=cli):
        with patch("teleop_xr.ros2.__main__.TeleopNode", return_value=node):
            mock_teleop_inst = mock_teleop.return_value
            main()

            callback = mock_teleop_inst.subscribe.call_args[0][0]

            xr_state = {
                "timestamp_unix_ms": 1000,
                "fetch_latency_ms": 10.0,
                "devices": [
                    {
                        "role": "hand",
                        "handedness": "right",
                        "joints": {
                            "wrist": {
                                "position": {"x": 0, "y": 0, "z": 0},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                            "thumb-tip": {
                                "position": {"x": 0, "y": 0, "z": 0},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        },
                    },
                    {
                        "role": "controller",
                        "handedness": "left",
                        "gripPose": {
                            "position": {"x": 0, "y": 0, "z": 0},
                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                        },
                        "gamepad": {
                            "buttons": [
                                {"pressed": True, "touched": True, "value": 1.0}
                            ],
                            "axes": [0.0, 0.0],
                        },
                    },
                ],
            }

            callback({}, xr_state)

            node.create_publisher.assert_any_call(ANY, "xr/hand_right/joints", 1)
            node.create_publisher.assert_any_call(ANY, "xr/controller_left/joy", 1)
