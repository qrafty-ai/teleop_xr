import os
import logging
import asyncio
import numpy as np
import tyro
from dataclasses import dataclass
from typing import Any

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings, RobotVisConfig, InputMode
from teleop_xr.messages import XRState
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.controller import IKController


@dataclass
class IKDemoCLI:
    """CLI options for the TeleopXR IK demo."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""
    port: int = 4443
    """Port to bind the server to."""


def main():
    # Use tyro for CLI parsing, consistent with other demos in the project
    cli = tyro.cli(IKDemoCLI)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo_ik")

    # 1. Initialize Robot, Solver, and Controller
    logger.info("Initializing Unitree H1 robot and IK solver...")
    robot = UnitreeH1Robot()
    solver = PyrokiSolver(robot)
    controller = IKController(robot, solver)

    # State container to track current joint configuration across updates
    state_container: dict[str, np.ndarray] = {"q": np.array(robot.get_default_config())}

    # 2. Configure Teleop with Visualization
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assets for H1_2 are located in teleop_xr/assets/h1_2
    urdf_path = os.path.join(current_dir, "assets", "h1_2", "h1_2.urdf")
    mesh_path = os.path.join(current_dir, "assets", "h1_2")

    if not os.path.exists(urdf_path):
        logger.warning(f"URDF not found at {urdf_path}. Visualization might not work.")
        robot_vis = None
    else:
        robot_vis = RobotVisConfig(urdf_path=urdf_path, mesh_path=mesh_path)

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        robot_vis=robot_vis,
        input_mode=InputMode.CONTROLLER,
    )

    teleop = Teleop(settings=settings)

    # 3. Define the XR Update Callback
    def on_xr_update(_pose: np.ndarray, message: dict[str, Any]):
        """
        Callback triggered on every XR state update.

        Args:
            _pose: Calculated end-effector pose (unused here, as IKController handles it).
            message: Raw XR state message data.
        """
        try:
            # The instruction says: Convert it: state = XRState.model_validate(message["data"])
            # In Teleop implementation, message passed to callback is the XR state data dict.
            # We use a robust approach to handle both nested and flat message formats.
            xr_data = message.get("data", message)
            state = XRState.model_validate(xr_data)

            # 4. Call controller.step(state, current_config)
            current_config = state_container["q"]
            new_config = np.array(controller.step(state, current_config))

            # 5. If output is new config, call teleop.publish_joint_state(joint_dict)
            if not np.array_equal(new_config, current_config):
                state_container["q"] = new_config

                # Convert to dict of name: value for the frontend visualization
                joint_dict = {
                    name: float(val)
                    for name, val in zip(robot.robot.joints.actuated_names, new_config)
                }

                # publish_joint_state is async, we call it from the sync callback using the event loop
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(teleop.publish_joint_state(joint_dict))
                except RuntimeError:
                    # Fallback if no loop is running in this thread
                    pass

        except Exception:
            # Silent fail for the high-frequency loop to avoid log spam during teleop
            pass

    # 6. Subscribe and Run
    teleop.subscribe(on_xr_update)

    logger.info(f"Starting IK Demo on {cli.host}:{cli.port}")
    logger.info("Open the WebXR app and hold BOTH grip buttons to engage IK control.")

    try:
        # 7. Run the server
        teleop.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
