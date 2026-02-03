import threading
import json
import time
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass, field, asdict
import cv2
import numpy as np
import tyro
import jax
from teleop_xr import Teleop, TF_RUB2FLU
from teleop_xr.video_stream import ExternalVideoSource
from teleop_xr.config import TeleopSettings
from teleop_xr.common_cli import CommonCLI
from teleop_xr.messages import XRState
from teleop_xr.events import EventProcessor, EventSettings, ButtonEvent, XRButton
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.controller import IKController
import transforms3d as t3d

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseStamped, PoseArray, TransformStamped
    from sensor_msgs.msg import Joy, Image, CompressedImage, JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Float64, String
    from tf2_ros import TransformBroadcaster
    from builtin_interfaces.msg import Time, Duration

    try:
        from cv_bridge import CvBridge

        HAS_CV_BRIDGE = True
    except ImportError:
        HAS_CV_BRIDGE = False
except ImportError:
    raise ImportError(
        "ROS2 is not sourced. Please source ROS2 before running this script."
    )

XR_HAND_JOINTS = [
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
]


def pose_dict_to_matrix(pose):
    if not pose or "position" not in pose or "orientation" not in pose:
        return None
    pos = pose["position"]
    quat = pose["orientation"]
    mat = t3d.affines.compose(
        [pos["x"], pos["y"], pos["z"]],
        t3d.quaternions.quat2mat([quat["w"], quat["x"], quat["y"], quat["z"]]),
        [1.0, 1.0, 1.0],
    )
    return TF_RUB2FLU @ mat


def matrix_to_pose_msg(mat):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(mat[0, 3]),
        float(mat[1, 3]),
        float(mat[2, 3]),
    )
    quat = t3d.quaternions.mat2quat(mat[:3, :3])
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )
    return pose


def ms_to_time(ms):
    sec = int(ms / 1000)
    nanosec = int((ms % 1000) * 1e6)
    return Time(sec=sec, nanosec=nanosec)


class ROSImageToVideoSource:
    def __init__(self, node, topic, source: ExternalVideoSource):
        self.node = node
        self.source = source
        self.bridge = CvBridge() if HAS_CV_BRIDGE else None

        if topic.endswith("/compressed"):
            self.sub = node.create_subscription(
                CompressedImage, topic, self.callback, 10
            )
        else:
            self.sub = node.create_subscription(Image, topic, self.callback, 10)

    def callback(self, msg):
        if isinstance(msg, CompressedImage):
            if not HAS_CV_BRIDGE:
                self.node.get_logger().error(
                    "cv_bridge not available, cannot decode CompressedImage"
                )
                return
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        elif isinstance(msg, Image):
            if HAS_CV_BRIDGE:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                # Basic ROS image to numpy conversion (OpenCV BGR compatible)
                # Assume rgb8 or bgr8 for now, but handle basic cases
                dtype = np.uint8
                if msg.encoding == "rgb8":
                    frame_rgb = np.frombuffer(msg.data, dtype=dtype).reshape(
                        msg.height, msg.width, 3
                    )
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                elif msg.encoding == "bgr8":
                    frame = np.frombuffer(msg.data, dtype=dtype).reshape(
                        msg.height, msg.width, 3
                    )
                elif msg.encoding == "mono8":
                    frame = np.frombuffer(msg.data, dtype=dtype).reshape(
                        msg.height, msg.width, 1
                    )
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    self.node.get_logger().warning(
                        f"Unsupported image encoding: {msg.encoding}"
                    )
                    return
        else:
            self.node.get_logger().error(f"Unknown message type: {type(msg)}")
            return

        self.source.put_frame(frame)


def build_joy(gamepad):
    if not gamepad:
        return None, None
    buttons_list = gamepad.get("buttons", [])
    buttons = [1 if b.get("pressed") else 0 for b in buttons_list]
    axes = list(gamepad.get("axes", [])) + [
        float(b.get("value", 0.0)) for b in buttons_list
    ]
    touched = [1 if b.get("touched") else 0 for b in buttons_list]
    return (buttons, axes), touched


@dataclass
class Ros2CLI(CommonCLI):
    mode: Literal["teleop", "ik"] = "teleop"
    """Operation mode: 'teleop' for standard ROS2 streaming, 'ik' for IK-based control."""

    # Explicit topics
    head_topic: Optional[str] = None
    wrist_left_topic: Optional[str] = None
    wrist_right_topic: Optional[str] = None

    # Custom streams
    extra_streams: Dict[str, str] = field(default_factory=dict)

    frame_id: str = "xr_local"
    publish_hand_tf: bool = False

    # ROS args (passed as remainder, but Tyro can capture list if explicit)
    # We will use this to pass args to rclpy
    ros_args: List[str] = field(default_factory=list)


class IKWorker(threading.Thread):
    """
    Dedicated worker thread for IK calculations.
    Consumes the latest available XRState and processes it.
    Publishes JointTrajectory messages to ROS2.
    """

    def __init__(
        self,
        controller: IKController,
        robot: UnitreeH1Robot,
        publisher: "rclpy.publisher.Publisher",
        state_container: dict,
        node: "rclpy.node.Node",
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.robot = robot
        self.publisher = publisher
        self.state_container = state_container
        self.node = node
        self.latest_xr_state: Optional[XRState] = None
        self.new_state_event = threading.Event()
        self.running = True

    def update_state(self, state: XRState):
        """Thread-safe update of the latest state."""
        self.latest_xr_state = state
        self.new_state_event.set()

    def run(self):
        while self.running:
            # Wait for new data
            if not self.new_state_event.wait(timeout=0.1):
                continue

            # Clear event immediately so we can detect new updates during processing
            self.new_state_event.clear()

            # Grab the latest state (atomic assignment in Python)
            state = self.latest_xr_state
            if state is None:
                continue

            try:
                current_config = self.state_container["q"]
                was_active = self.controller.active

                t0 = time.perf_counter()
                new_config = np.array(self.controller.step(state, current_config))
                dt = time.perf_counter() - t0

                self.state_container["solve_time"] = dt
                self.state_container["active"] = self.controller.active
                is_active = self.controller.active

                if not was_active and is_active:
                    self.node.get_logger().info("IK engagement started")

                if not np.array_equal(new_config, current_config):
                    self.state_container["q"] = new_config

                    # Publish to ROS2
                    msg = JointTrajectory()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.joint_names = self.robot.robot.joints.actuated_names

                    point = JointTrajectoryPoint()
                    point.positions = [float(val) for val in new_config]
                    point.time_from_start = Duration(sec=0, nanosec=int(1e7))  # 10ms
                    msg.points = [point]

                    self.publisher.publish(msg)

            except Exception as e:
                self.node.get_logger().error(f"Error in IK Worker: {e}")


def main():
    jax.config.update("jax_platform_name", "cpu")
    cli = tyro.cli(Ros2CLI)

    rclpy.init(args=["--ros-args"] + cli.ros_args)
    node = rclpy.create_node("teleop")

    # --- Mode Setup ---
    robot = None
    solver = None
    controller = None
    ik_worker = None
    state_container: dict[str, Any] = {
        "active": False,
        "solve_time": 0.0,
        "xr_state": None,
    }

    if cli.mode == "ik":
        node.get_logger().info("Initializing Unitree H1 robot and IK solver...")
        robot = UnitreeH1Robot()
        solver = PyrokiSolver(robot)
        controller = IKController(robot, solver)
        state_container["q"] = np.array(robot.get_default_config())

        # ROS2 Pub/Sub for IK
        ik_pub = node.create_publisher(JointTrajectory, "/joint_trajectory", 10)

        def joint_state_callback(msg: JointState):
            # Only update q from robot if not actively engaging IK or if we want to follow real robot
            # For now, we update q if not active to avoid drift before starting
            if not state_container.get("active", False):
                current_q = state_container["q"].copy()
                actuated_names = robot.robot.joints.actuated_names
                for i, name in enumerate(actuated_names):
                    if name in msg.name:
                        idx = msg.name.index(name)
                        current_q[i] = msg.position[idx]
                state_container["q"] = current_q

        node.create_subscription(JointState, "/joint_states", joint_state_callback, 10)
        ik_worker = IKWorker(controller, robot, ik_pub, state_container, node)
        ik_worker.start()
    else:
        node.get_logger().info("ROS2 Node starting in Teleop mode")

    # Merge topics
    topics = {}
    if cli.head_topic:
        topics["head"] = cli.head_topic
    if cli.wrist_left_topic:
        topics["wrist_left"] = cli.wrist_left_topic
    if cli.wrist_right_topic:
        topics["wrist_right"] = cli.wrist_right_topic
    topics.update(cli.extra_streams)

    video_sources = {}
    for key, topic in topics.items():
        source = ExternalVideoSource()
        ROSImageToVideoSource(node, topic, source)
        video_sources[key] = source

    # Create config dict for camera views
    camera_views = {k: {"device": topic} for k, topic in topics.items()}

    robot_vis = None
    if cli.mode == "ik" and robot:
        robot_vis = robot.get_vis_config()

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        input_mode=cli.input_mode,
        camera_views=camera_views,
        robot_vis=robot_vis,
    )

    teleop = Teleop(
        settings=settings,
        video_sources=video_sources,
    )
    broadcaster = TransformBroadcaster(node)

    publishers = {}

    def get_publisher(msg_type, topic):
        if topic not in publishers:
            publishers[topic] = node.create_publisher(msg_type, topic, 1)
        return publishers[topic]

    def publish_pose(topic, pose_dict, stamp, child_frame_id, tf_stamp):
        if not pose_dict:
            return
        mat = pose_dict_to_matrix(pose_dict)
        if mat is None:
            return
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = cli.frame_id
        msg.pose = matrix_to_pose_msg(mat)
        get_publisher(PoseStamped, topic).publish(msg)

        tf = TransformStamped()
        tf.header.stamp = tf_stamp  # Use PC timestamp for TF
        tf.header.frame_id = cli.frame_id
        tf.child_frame_id = child_frame_id
        tf.transform.translation.x = msg.pose.position.x
        tf.transform.translation.y = msg.pose.position.y
        tf.transform.translation.z = msg.pose.position.z
        tf.transform.rotation = msg.pose.orientation
        broadcaster.sendTransform(tf)

    def publish_joy(topic, joy_data, stamp):
        buttons, axes = joy_data
        msg = Joy()
        msg.header.stamp = stamp
        msg.header.frame_id = cli.frame_id
        msg.buttons = buttons
        msg.axes = axes
        get_publisher(Joy, topic).publish(msg)

    def publish_hand(device, stamp):
        handed = device.get("handedness", "none")
        joints_dict = device.get("joints", {})
        if not joints_dict:
            return

        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = cli.frame_id

        for joint_name in XR_HAND_JOINTS:
            joint_pose_dict = joints_dict.get(joint_name)
            mat = pose_dict_to_matrix(joint_pose_dict)
            if mat is None:
                continue
            pose_msg = matrix_to_pose_msg(mat)
            pose_array.poses.append(pose_msg)

            if cli.publish_hand_tf:
                tf = TransformStamped()
                tf.header.stamp = stamp
                tf.header.frame_id = cli.frame_id
                tf.child_frame_id = f"xr/hand_{handed}/{joint_name}"
                tf.transform.translation.x = pose_msg.position.x
                tf.transform.translation.y = pose_msg.position.y
                tf.transform.translation.z = pose_msg.position.z
                tf.transform.rotation = pose_msg.orientation
                broadcaster.sendTransform(tf)

        get_publisher(PoseArray, f"xr/hand_{handed}/joints").publish(pose_array)

    event_processor = EventProcessor(EventSettings())

    from teleop_xr.events import ButtonEventType

    def publish_button_event(event: ButtonEvent):
        try:
            # Robot Reset: Double-press on deadman switch (SQUEEZE)
            if (
                event.type == ButtonEventType.DOUBLE_PRESS
                and event.button == XRButton.SQUEEZE
            ):
                if cli.mode == "ik" and robot and ik_worker and controller:
                    default_q = np.array(robot.get_default_config())
                    state_container["q"] = default_q
                    controller.reset()
                    node.get_logger().info(
                        "Resetting Robot Joint State and IK Snapshots"
                    )

            msg = String()
            msg.data = json.dumps(asdict(event))
            get_publisher(String, f"xr/events/{event.type.value}").publish(msg)
        except Exception as exc:
            node.get_logger().warning(f"publish_button_event error: {exc}")

    event_processor.on_button_down(callback=publish_button_event)
    event_processor.on_button_up(callback=publish_button_event)
    event_processor.on_double_press(callback=publish_button_event)
    event_processor.on_long_press(callback=publish_button_event)

    def teleop_xr_state_callback(_pose, xr_state):
        try:
            # Process events
            event_processor.process(_pose, xr_state)

            # Parse state for IK and container
            xr_data = xr_state.get("data", xr_state)
            state = XRState.model_validate(xr_data)
            state_container["xr_state"] = state

            if ik_worker:
                ik_worker.update_state(state)

            ms = xr_state.get("timestamp_unix_ms") if xr_state else None
            stamp = (
                ms_to_time(ms) if ms is not None else node.get_clock().now().to_msg()
            )
            # Use PC timestamp for TF to avoid lag from headset/PC time difference
            tf_stamp = node.get_clock().now().to_msg()

            # Publish fetch latency if available
            fetch_latency = xr_state.get("fetch_latency_ms") if xr_state else None
            if fetch_latency is not None:
                latency_msg = Float64()
                latency_msg.data = float(fetch_latency)
                get_publisher(Float64, "xr/fetch_latency_ms").publish(latency_msg)

            for device in xr_state.get("devices", []) if xr_state else []:
                role = device.get("role")
                handed = device.get("handedness", "none")

                if role == "head":
                    publish_pose(
                        "xr/head/pose",
                        device.get("pose"),
                        stamp,
                        "xr/head",
                        tf_stamp,
                    )

                if role == "controller":
                    publish_pose(
                        f"xr/controller_{handed}/pose",
                        device.get("gripPose"),
                        stamp,
                        f"xr/controller_{handed}/pose",
                        tf_stamp,
                    )

                    joy_payload, touched = build_joy(device.get("gamepad"))
                    if joy_payload:
                        publish_joy(f"xr/controller_{handed}/joy", joy_payload, stamp)
                    if touched:
                        publish_joy(
                            f"xr/controller_{handed}/joy_touched",
                            (touched, []),
                            stamp,
                        )

                if role == "hand":
                    publish_hand(device, stamp)
        except Exception as exc:
            node.get_logger().warning(f"xr_state callback error: {exc}")

    teleop.subscribe(teleop_xr_state_callback)

    # Start ROS spin thread
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    teleop.run()


if __name__ == "__main__":
    main()
