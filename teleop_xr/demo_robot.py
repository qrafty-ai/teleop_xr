import os
import math
import time
import asyncio
import threading
import logging
from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings, RobotVisConfig


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo_robot")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir))

    urdf_path = os.path.join(repo_root, "tests", "fixtures", "demo_openarm.urdf")
    mesh_path = os.path.join(repo_root, "tests", "fixtures")

    if not os.path.exists(urdf_path):
        logger.error(f"URDF path does not exist: {urdf_path}")
        return

    logger.info(f"Using URDF: {urdf_path}")
    logger.info(f"Using Mesh Path: {mesh_path}")

    robot_vis = RobotVisConfig(urdf_path=urdf_path, mesh_path=mesh_path)
    settings = TeleopSettings(host="0.0.0.0", port=4443, robot_vis=robot_vis)

    teleop = Teleop(settings=settings)

    async def oscillate_joints():
        logger.info("Starting joint oscillation background task")
        while True:
            t = time.time()
            joints = {"joint1": math.sin(t), "joint2": math.cos(t)}
            try:
                await teleop.publish_joint_state(joints)
            except Exception as e:
                logger.error(f"Error publishing joint state: {e}")
            await asyncio.sleep(0.05)

    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(oscillate_joints())
        except Exception as e:
            logger.error(f"Oscillation thread error: {e}")

    osc_thread = threading.Thread(target=thread_target, daemon=True)
    osc_thread.start()

    logger.info("Starting Teleop server on port 4443...")
    try:
        teleop.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
