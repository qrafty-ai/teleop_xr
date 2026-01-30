import threading
from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import tyro
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings, ViewConfig
from teleop_xr.ik.mink_solver import MinkIKSolver
from teleop_xr.ros2.__main__ import ROSImageToVideoSource, Ros2CLI
from teleop_xr.video_stream import ExternalVideoSource

try:
    from robot_descriptions import g1_mj_description

    DEFAULT_MODEL = g1_mj_description.MJCF_PATH
except ImportError:
    DEFAULT_MODEL = ""


@dataclass
class Ros2IKCLI(Ros2CLI):
    model: str = DEFAULT_MODEL
    joint_topic: str = "/joint_trajectory"
    joint_state_topic: str = "/joint_states"
    head_frame: str = "head_link"
    left_hand_frame: str = "left_palm"
    right_hand_frame: str = "right_palm"
    viewer: bool = False
    rate: int = 100


class IKNode(Node):
    def __init__(self, cli: Ros2IKCLI):
        super().__init__("teleop_ik")
        self.cli = cli
        self.joint_names: list[str] = []
        self.current_q: np.ndarray | None = None
        self.lock = threading.Lock()

        self.sub = self.create_subscription(
            JointState, cli.joint_state_topic, self.joint_state_callback, 10
        )
        self.pub = self.create_publisher(JointTrajectory, cli.joint_topic, 10)

    def joint_state_callback(self, msg: JointState):
        with self.lock:
            self.joint_names = list(msg.name)
            self.current_q = np.array(msg.position)

    def publish_trajectory(self, q: np.ndarray, joint_names: list[str]):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        # Set a small time from start to indicate immediate execution
        point.time_from_start.nanosec = int(1e9 / self.cli.rate)
        msg.points.append(point)
        self.pub.publish(msg)


def main():
    cli = tyro.cli(Ros2IKCLI)

    rclpy.init(args=["--ros-args"] + cli.ros_args)
    node = IKNode(cli)

    end_effector_frames = {
        "head": cli.head_frame,
        "left_hand": cli.left_hand_frame,
        "right_hand": cli.right_hand_frame,
    }

    task_weights = {
        "head": 0.5,
        "left_hand": 1.0,
        "right_hand": 1.0,
    }

    solver = MinkIKSolver(
        model_path=cli.model,
        end_effector_frames=end_effector_frames,
        task_weights=task_weights,
    )
    solver.dt = 1.0 / cli.rate

    # Video streaming setup (reusing ROSImageToVideoSource from ros2.__main__)
    topics: dict[str, str] = {}
    if cli.head_topic:
        topics["head"] = cli.head_topic
    if cli.wrist_left_topic:
        topics["wrist_left"] = cli.wrist_left_topic
    if cli.wrist_right_topic:
        topics["wrist_right"] = cli.wrist_right_topic
    topics.update(cli.extra_streams)

    video_sources: dict[str, ExternalVideoSource] = {}
    for key, topic in topics.items():
        source = ExternalVideoSource()
        ROSImageToVideoSource(node, topic, source)
        video_sources[key] = source

    camera_views = {k: ViewConfig(device=topic) for k, topic in topics.items()}

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        input_mode=cli.input_mode,
        camera_views=camera_views,
        multi_eef_mode=True,
    )

    teleop = Teleop(
        settings=settings,
        video_sources=video_sources,  # type: ignore
    )

    viewer = None
    if cli.viewer:
        viewer = mujoco.viewer.launch_passive(solver.model, solver.data)

    def teleop_callback(targets: dict[str, np.ndarray], _: dict[str, Any]):
        with node.lock:
            if node.current_q is None:
                # Use solver's internal state if we haven't received joint states yet
                current_q = solver.data.qpos.copy()
            else:
                # Map received joint states to solver's qpos
                # MinkIKSolver expects qpos in the order of the MuJoCo model
                current_q = solver.data.qpos.copy()
                for i, name in enumerate(node.joint_names):
                    joint_id = mujoco.mj_name2id(
                        solver.model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    if joint_id != -1:
                        qpos_adr = solver.model.jnt_qposadr[joint_id]
                        if node.current_q is not None:
                            current_q[qpos_adr] = node.current_q[i]

        new_q = solver.solve(targets, current_q)

        # Update MuJoCo for viewer/internal state
        solver.data.qpos[:] = new_q
        mujoco.mj_forward(solver.model, solver.data)

        if viewer is not None:
            solver.update_viewer(viewer)

        # Publish resulting q
        # Extract active joint names from MuJoCo model
        active_joint_names: list[str] = []
        active_q: list[float] = []
        for i in range(solver.model.njnt):
            name = mujoco.mj_id2name(solver.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if solver.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                # Skip free joint (base) for JointTrajectory
                continue
            active_joint_names.append(name)
            # JointTrajectory expects a single float per joint name.
            # This assumes 1-DoF joints (hinge/slide).
            active_q.append(float(new_q[solver.model.jnt_qposadr[i]]))

        node.publish_trajectory(np.array(active_q), active_joint_names)

    teleop.subscribe(teleop_callback)

    # Start ROS spin thread
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
