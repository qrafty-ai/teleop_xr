from teleop import Teleop
import argparse
import threading

try:
    import rclpy
    from std_msgs.msg import String
except ImportError:
    raise ImportError(
        "ROS2 is not sourced. Please source ROS2 before running this script."
    )

try:
    from teleop.utils.jacobi_robot_ros import JacobiRobotROS
except ImportError:
    raise ImportError(
        "JacobiRobotROS is not available. Please install the teleop with [utils] extra."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--omit-current-pose", action="store_true", help="Omit usage of current pose"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=4443, help="Port number")
    parser.add_argument(
        "--natural-position",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Natural position of the phone",
    )
    parser.add_argument(
        "--natural-orientation",
        nargs=3,
        type=float,
        default=[0.0, -45, 0.0],
        help="Natural orientation of the phone (in degrees)",
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="end_effector",
        help="End effector name (e.g., 'panda_hand')",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=None,
        help="List of joint names",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="controller",
        choices=["controller", "hand", "auto"],
        help="Input mode for the teleop (default: controller)",
    )
    parser.add_argument(
        "--ros-args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to ROS",
        default=[],
    )

    args = parser.parse_args()

    rclpy.init(args=["--ros-args"] + args.ros_args)

    node = rclpy.create_node("teleop")
    gripper_publisher = node.create_publisher(String, "/gripper_command", 1)
    teleop = Teleop(
        host=args.host,
        port=args.port,
        natural_phone_orientation_euler=args.natural_orientation,
        natural_phone_position=args.natural_position,
        input_mode=args.input_mode,
    )
    robot = JacobiRobotROS(
        node,
        ee_link=args.ee_link,
        joint_names=args.joint_names,
    )
    robot.reset_joint_states()
    ee_pose = robot.get_ee_pose()
    teleop.set_pose(ee_pose)

    def teleop_pose_callback(pose, xr_state):
        nonlocal teleop
        nonlocal node
        nonlocal robot

        # Helper to select controller device from xr_state (prefer right, fallback left)
        devices = xr_state.get("devices", [])
        controller = next(
            (
                d
                for d in devices
                if d.get("role") == "controller" and d.get("handedness") == "right"
            ),
            None,
        )
        if not controller:
            controller = next(
                (
                    d
                    for d in devices
                    if d.get("role") == "controller" and d.get("handedness") == "left"
                ),
                None,
            )

        if not controller:
            return

        buttons = controller.get("gamepad", {}).get("buttons", [])
        # Map move gating to controller trigger: buttons[0].pressed (or value > 0.5 if present)
        move = False
        if len(buttons) > 0:
            move = (
                buttons[0].get("pressed", False) or buttons[0].get("value", 0.0) > 0.5
            )

        # Map gripper to controller grip: buttons[1] value/pressed; output string "close" if grip > 0.5 else "open"
        gripper = "open"
        if len(buttons) > 1:
            grip_val = buttons[1].get(
                "value", 1.0 if buttons[1].get("pressed", False) else 0.0
            )
            gripper = "close" if grip_val > 0.5 else "open"

        gripper_publisher.publish(String(data=gripper))

        if not robot.are_joint_states_received():
            return

        if not move:
            return

        robot.servo_to_pose(pose, 0.2)

    teleop.subscribe(teleop_pose_callback)

    # start the ros2 node in a separate thread
    t = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )
    t.start()

    teleop.run()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
