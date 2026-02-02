import os
import logging
import time
import asyncio
import threading
import numpy as np
import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass
from typing import Any, Deque, Optional
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box

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


class TUIHandler(logging.Handler):
    """Custom logging handler to send logs to a deque for TUI display."""

    def __init__(self, log_queue: Deque[str]):
        super().__init__()
        self.log_queue = log_queue
        self.formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%H:%M:%S"
        )

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_queue.append(msg)
        except Exception:
            self.handleError(record)


def generate_status_table(
    active: bool,
    solve_time: float,
    parse_time: float,
    xr_state: XRState | None,
    controller: IKController,
    robot: UnitreeH1Robot,
    current_q: np.ndarray,
) -> Panel:
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    status_style = "bold green" if active else "dim yellow"
    status_text = "ACTIVE" if active else "IDLE (Hold Grip)"
    table.add_row("IK Status", f"[{status_style}]{status_text}[/{status_style}]")
    table.add_row("Solve Time", f"{solve_time * 1000:.2f} ms")
    table.add_row("Parse Time", f"{parse_time * 1000:.2f} ms")

    if active and xr_state:
        curr_poses = controller._get_device_poses(xr_state)
        snap_poses = controller.snapshot_xr

        current_fk = robot.forward_kinematics(jnp.array(current_q))

        if "right" in curr_poses:
            t_ctrl_r = curr_poses["right"].translation()
            table.add_row(
                "Right Controller Pos",
                f"x={t_ctrl_r[0]:.3f} y={t_ctrl_r[1]:.3f} z={t_ctrl_r[2]:.3f}",
            )

        if "right" in current_fk:
            t_robot_r = current_fk["right"].translation()
            table.add_row(
                "Right Robot Hand Pos",
                f"x={t_robot_r[0]:.3f} y={t_robot_r[1]:.3f} z={t_robot_r[2]:.3f}",
            )

        table.add_section()

        for hand in ["left", "right"]:
            if hand in curr_poses and hand in snap_poses:
                t_curr = curr_poses[hand].translation()
                t_init = snap_poses[hand].translation()
                delta = t_curr - t_init
                table.add_row(
                    f"{hand.title()} Delta (XR)",
                    f"x={delta[0]:.3f} y={delta[1]:.3f} z={delta[2]:.3f}",
                )

                if abs(delta[1]) > 0.1:
                    table.add_row(
                        f"Alignment Check {hand}",
                        "[bold red]!! Moving Y (Up) in XR -> Check Robot Z !!",
                    )

    return Panel(table, title="[bold]System Status[/bold]", border_style="blue")


def generate_log_panel(log_queue: Deque[str]) -> Panel:
    return Panel(
        Group(*[str(m) for m in list(log_queue)[-15:]]),
        title="[bold]Logs[/bold]",
        border_style="white",
        box=box.ROUNDED,
    )


class IKWorker(threading.Thread):
    """
    Dedicated worker thread for IK calculations.
    Consumes the latest available XRState and processes it.
    """

    def __init__(
        self,
        controller: IKController,
        robot: UnitreeH1Robot,
        teleop: Teleop,
        state_container: dict,
        logger: logging.Logger,
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.robot = robot
        self.teleop = teleop
        self.state_container = state_container
        self.logger = logger
        self.latest_xr_state: Optional[XRState] = None
        self.new_state_event = threading.Event()
        self.running = True
        self.teleop_loop = None  # Will be set when on_xr_update runs

    def update_state(self, state: XRState):
        """Thread-safe update of the latest state."""
        self.latest_xr_state = state
        self.new_state_event.set()

    def set_teleop_loop(self, loop: asyncio.AbstractEventLoop):
        if self.teleop_loop is None:
            self.teleop_loop = loop

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
                    self.logger.info("in_control start - Taking Snapshots")
                    self.logger.info(
                        f"Init XR: {list(self.controller.snapshot_xr.keys())}"
                    )

                if not np.array_equal(new_config, current_config):
                    self.state_container["q"] = new_config
                    joint_dict = {
                        name: float(val)
                        for name, val in zip(
                            self.robot.robot.joints.actuated_names, new_config
                        )
                    }

                    if self.teleop_loop and self.teleop_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.teleop.publish_joint_state(joint_dict),
                            self.teleop_loop,
                        )

            except Exception as e:
                self.logger.error(f"Error in IK Worker: {e}")


def main():
    jax.config.update("jax_platform_name", "cpu")

    cli = tyro.cli(IKDemoCLI)

    log_queue: Deque[str] = deque(maxlen=50)
    logging.basicConfig(level=logging.INFO, handlers=[TUIHandler(log_queue)])
    logging.getLogger("jaxls").setLevel(logging.WARNING)
    logger = logging.getLogger("demo_ik")

    logger.info("Initializing Unitree H1 robot and IK solver...")
    robot = UnitreeH1Robot()
    solver = PyrokiSolver(robot)
    controller = IKController(robot, solver)

    state_container: dict[str, Any] = {
        "q": np.array(robot.get_default_config()),
        "active": False,
        "solve_time": 0.0,
        "parse_time": 0.0,
        "xr_state": None,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "assets", "h1_2", "h1_2.urdf")
    mesh_path = os.path.join(current_dir, "assets", "h1_2")

    if not os.path.exists(urdf_path):
        logger.warning(f"URDF not found at {urdf_path}")
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

    # Initialize IK Worker
    ik_worker = IKWorker(controller, robot, teleop, state_container, logger)
    ik_worker.start()

    def on_xr_update(_pose: np.ndarray, message: dict[str, Any]):
        try:
            # Capture the event loop of the thread running Teleop
            try:
                loop = asyncio.get_running_loop()
                ik_worker.set_teleop_loop(loop)
            except RuntimeError:
                pass

            t_parse_start = time.perf_counter()
            xr_data = message.get("data", message)
            state = XRState.model_validate(xr_data)

            # Fast update: just push data to worker
            state_container["parse_time"] = time.perf_counter() - t_parse_start
            state_container["xr_state"] = state

            ik_worker.update_state(state)

        except Exception:
            # logger.error(f"Error in parsing: {e}")
            pass

    teleop.subscribe(on_xr_update)

    console = Console()
    layout = Layout()
    layout.split_row(Layout(name="status", ratio=1), Layout(name="logs", ratio=1))

    logger.info(f"Starting IK Demo on {cli.host}:{cli.port}")
    logger.info("Open WebXR app and hold BOTH grips to engage.")

    try:
        import threading

        t = threading.Thread(target=teleop.run, daemon=True)
        t.start()

        with Live(layout, refresh_per_second=10, console=console):
            while t.is_alive():
                layout["status"].update(
                    generate_status_table(
                        state_container["active"],
                        state_container["solve_time"],
                        state_container["parse_time"],
                        state_container["xr_state"],
                        controller,
                        robot,
                        state_container["q"],
                    )
                )
                layout["logs"].update(generate_log_panel(log_queue))
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        ik_worker.running = False
        ik_worker.join()


if __name__ == "__main__":
    main()
