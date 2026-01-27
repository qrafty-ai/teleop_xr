import argparse
from teleop import Teleop, TF_RUB2FLU
import transforms3d as t3d

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseStamped, PoseArray, TransformStamped
    from sensor_msgs.msg import Joy
    from tf2_ros import TransformBroadcaster
    from builtin_interfaces.msg import Time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=4443, help="Port number")
    parser.add_argument(
        "--input-mode",
        choices=["controller", "hand", "auto"],
        default="controller",
        help="Input mode for XR state",
    )
    parser.add_argument(
        "--frame-id", default="xr_local", help="Fixed frame ID for XR poses"
    )
    parser.add_argument(
        "--publish-hand-tf", action="store_true", help="Publish TF for hand joints"
    )
    parser.add_argument(
        "--ros-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to ROS",
        default=[],
    )

    args = parser.parse_args()

    rclpy.init(args=["--ros-args"] + args.ros_args)

    teleop = Teleop(
        host=args.host,
        port=args.port,
        input_mode=args.input_mode,
    )
    node = rclpy.create_node("teleop")
    broadcaster = TransformBroadcaster(node)

    publishers = {}

    def get_publisher(msg_type, topic):
        if topic not in publishers:
            publishers[topic] = node.create_publisher(msg_type, topic, 1)
        return publishers[topic]

    def publish_pose(topic, pose_dict, stamp, child_frame_id):
        if not pose_dict:
            return
        mat = pose_dict_to_matrix(pose_dict)
        if mat is None:
            return
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = args.frame_id
        msg.pose = matrix_to_pose_msg(mat)
        get_publisher(PoseStamped, topic).publish(msg)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = args.frame_id
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
        msg.header.frame_id = args.frame_id
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
        pose_array.header.frame_id = args.frame_id

        for joint_name in XR_HAND_JOINTS:
            joint_pose_dict = joints_dict.get(joint_name)
            mat = pose_dict_to_matrix(joint_pose_dict)
            if mat is None:
                continue
            pose_msg = matrix_to_pose_msg(mat)
            pose_array.poses.append(pose_msg)

            if args.publish_hand_tf:
                tf = TransformStamped()
                tf.header.stamp = stamp
                tf.header.frame_id = args.frame_id
                tf.child_frame_id = f"xr/hand_{handed}/{joint_name}"
                tf.transform.translation.x = pose_msg.position.x
                tf.transform.translation.y = pose_msg.position.y
                tf.transform.translation.z = pose_msg.position.z
                tf.transform.rotation = pose_msg.orientation
                broadcaster.sendTransform(tf)

        get_publisher(PoseArray, f"xr/hand_{handed}/joints").publish(pose_array)

    def teleop_xr_state_callback(_pose, xr_state):
        try:
            ms = xr_state.get("timestamp_unix_ms") if xr_state else None
            stamp = (
                ms_to_time(ms) if ms is not None else node.get_clock().now().to_msg()
            )

            for device in xr_state.get("devices", []) if xr_state else []:
                role = device.get("role")
                handed = device.get("handedness", "none")

                if role == "head":
                    publish_pose(
                        "xr/head/pose",
                        device.get("pose"),
                        stamp,
                        "xr/head",
                    )

                if role == "controller":
                    publish_pose(
                        f"xr/controller_{handed}/target_ray",
                        device.get("targetRayPose"),
                        stamp,
                        f"xr/controller_{handed}/target_ray",
                    )
                    publish_pose(
                        f"xr/controller_{handed}/grip",
                        device.get("gripPose"),
                        stamp,
                        f"xr/controller_{handed}/grip",
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

        try:
            rclpy.spin_once(node, timeout_sec=0.0)
        except Exception as exc:
            node.get_logger().debug(f"spin_once failed: {exc}")

    teleop.subscribe(teleop_xr_state_callback)
    teleop.run()


if __name__ == "__main__":
    main()
