import logging
import time
import asyncio
import threading
import numpy as np
import jax
import jax.numpy as jnp
import tyro
from dataclasses import dataclass, field
from typing import Any, Deque, Optional, Union, Dict, Literal
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.common_cli import CommonCLI
from teleop_xr.messages import XRState
from teleop_xr.camera_views import build_camera_views_config
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.controller import IKController
from teleop_xr.events import (
    EventProcessor,
    EventSettings,
    ButtonEvent,
    ButtonEventType,
    XRButton,
)


# Maximum number of events to display in the event log
MAX_EVENT_LOG_SIZE = 10


@dataclass
class DemoCLI(CommonCLI):
    """CLI options for the unified TeleopXR demo."""

    mode: Literal["teleop", "ik"] = "teleop"
    """Operation mode: 'teleop' for visualization only, 'ik' for H1 robot control. (default: teleop)"""

    # Camera device configuration
    head_device: Union[int, str, None] = None
    """Camera device for head view (index or path)."""
    wrist_left_device: Union[int, str, None] = None
    """Camera device for left wrist view (index or path)."""
    wrist_right_device: Union[int, str, None] = None
    """Camera device for right wrist view (index or path)."""
    camera: Dict[str, Union[int, str]] = field(default_factory=dict)
    """Extra cameras: --camera key1 dev1 key2 dev2 (e.g., --camera left /dev/video4 right /dev/video6)"""

    no_tui: bool = False
    """Disable TUI for cleaner logging debugging"""

    # Event system configuration
    double_press_ms: float = 300
    """Maximum time between presses to count as double-press (ms)."""
    long_press_ms: float = 500
    """Minimum hold time to count as long-press (ms)."""
    enable_events: bool = True
    """Enable the event system for gesture detection."""


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


# --- Teleop TUI Helpers ---


def _get_event_style(event_type: ButtonEventType) -> str:
    """Return a rich style string for an event type."""
    styles = {
        ButtonEventType.BUTTON_DOWN: "green",
        ButtonEventType.BUTTON_UP: "yellow",
        ButtonEventType.DOUBLE_PRESS: "bold magenta",
        ButtonEventType.LONG_PRESS: "bold red",
    }
    return styles.get(event_type, "white")


def _get_event_icon(event_type: ButtonEventType) -> str:
    """Return an icon/symbol for an event type."""
    icons = {
        ButtonEventType.BUTTON_DOWN: "▼",
        ButtonEventType.BUTTON_UP: "▲",
        ButtonEventType.DOUBLE_PRESS: "⚡",
        ButtonEventType.LONG_PRESS: "⏳",
    }
    return icons.get(event_type, "•")


def generate_state_table(xr_state: Optional[XRState] = None) -> Table:
    """Generate a rich table showing XR device states."""
    table = Table(
        title="[bold cyan]XR Device State[/bold cyan]",
        box=box.ROUNDED,
        expand=True,
        title_justify="left",
    )
    table.add_column("Role", style="cyan", no_wrap=True, width=10)
    table.add_column("Hand", style="magenta", width=6)
    table.add_column("Position (x, y, z)", style="green", width=20)
    table.add_column("Orientation (x, y, z, w)", style="yellow", width=26)
    table.add_column("Inputs", style="blue")

    if not xr_state or not xr_state.devices:
        table.add_row("-", "-", "-", "-", "[dim]Waiting for data...[/dim]")
        return table

    devices = list(xr_state.devices)

    # Sort devices for stable order: head, left, right
    def sort_key(d):
        role = d.role.value if d.role else ""
        hand = d.handedness.value if d.handedness else ""
        priority = {"head": 0, "controller": 1, "hand": 2}
        hand_prio = {"left": 0, "right": 1, "none": 2}
        return (priority.get(role, 99), hand_prio.get(hand, 99))

    devices.sort(key=sort_key)

    for dev in devices:
        role = dev.role.value if dev.role else "unknown"
        hand = dev.handedness.value if dev.handedness else "none"

        # Parse Pose
        pose = dev.pose or dev.gripPose
        if pose:
            pos = pose.position
            ort = pose.orientation
            pos_str = (
                f"{pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f}"
            )
            ort_str = f"{ort.get('x', 0):.2f}, {ort.get('y', 0):.2f}, {ort.get('z', 0):.2f}, {ort.get('w', 1):.2f}"
        else:
            pos_str = "-"
            ort_str = "-"

        # Parse Inputs (Buttons/Axes)
        inputs_parts = []
        if dev.gamepad:
            buttons = dev.gamepad.buttons
            axes = dev.gamepad.axes

            pressed = [i for i, b in enumerate(buttons) if b.pressed]
            if pressed:
                inputs_parts.append(f"Btn:{pressed}")

            active_axes = [f"{i}:{v:.1f}" for i, v in enumerate(axes) if abs(v) > 0.1]
            if active_axes:
                inputs_parts.append(f"Ax:{','.join(active_axes)}")

        if dev.joints:
            inputs_parts.append(f"{len(dev.joints)} joints")

        inputs_str = " | ".join(inputs_parts) if inputs_parts else "-"

        table.add_row(role, hand, pos_str, ort_str, inputs_str)

    return table


def generate_event_panel(event_log: deque) -> Panel:
    """Generate a rich panel showing recent button events."""
    if not event_log:
        content = Text(
            "No events yet. Press buttons in VR!", justify="left", style="dim"
        )
    else:
        lines = []
        for event in reversed(event_log):
            icon = _get_event_icon(event.type)
            style = _get_event_style(event.type)
            event_name = event.type.value.replace("_", " ").title()
            controller = event.controller.value.upper()
            button = event.button.value.replace("_", " ").title()

            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(f"[{controller}] ", style="bold white")
            line.append(f"{button} ", style="cyan")
            line.append(f"{event_name}", style=style)

            if event.hold_duration_ms is not None:
                line.append(f" ({event.hold_duration_ms:.0f}ms)", style="dim")

            lines.append(line)

        content = Group(*lines)

    return Panel(
        content,
        title="[bold magenta]Event Log[/bold magenta]",
        title_align="left",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def generate_help_panel() -> Panel:
    """Generate a help panel showing event legend."""
    help_text = Text()
    help_text.append("▼ ", style="green")
    help_text.append("DOWN  ")
    help_text.append("▲ ", style="yellow")
    help_text.append("UP  ")
    help_text.append("⚡ ", style="bold magenta")
    help_text.append("DOUBLE  ")
    help_text.append("⏳ ", style="bold red")
    help_text.append("LONG")

    return Panel(
        help_text,
        title="[bold blue]Legend[/bold blue]",
        title_align="left",
        box=box.ROUNDED,
        padding=(0, 1),
    )


# --- IK TUI Helpers ---


def generate_ik_status_table(
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

    return Panel(table, title="[bold]IK Status[/bold]", border_style="blue")


def generate_ik_controls_panel() -> Panel:
    """Generate a panel showing IK key bindings."""
    text = Text()
    text.append("• Hold ", style="dim")
    text.append("BOTH GRIPS", style="bold yellow")
    text.append(" to engage IK control\n", style="dim")
    text.append("• Double-click ", style="dim")
    text.append("DEADMAN (Grip)", style="bold magenta")
    text.append(" to reset joints", style="dim")

    return Panel(
        text,
        title="[bold blue]IK Key Bindings[/bold blue]",
        title_align="left",
        box=box.ROUNDED,
        padding=(0, 1),
    )


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

    cli = tyro.cli(DemoCLI)

    # Backward compatibility: default to head on device 0 if no flags provided
    if (
        cli.head_device is None
        and cli.wrist_left_device is None
        and cli.wrist_right_device is None
        and not cli.camera
    ):
        cli.head_device = 0  # Default to index 0

    log_queue: Deque[str] = deque(maxlen=50)
    event_log: deque[ButtonEvent] = deque(maxlen=MAX_EVENT_LOG_SIZE)

    # Configure logging
    handlers = []
    if not cli.no_tui:
        handlers.append(TUIHandler(log_queue))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.getLogger("jaxls").setLevel(logging.WARNING)
    # Silence uvicorn access logs when TUI is active to prevent spam
    if not cli.no_tui:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    logger = logging.getLogger("demo")
    logger.info(
        f"Starting TeleopXR Demo in {cli.mode.upper()} mode on {cli.host}:{cli.port}"
    )

    # --- Mode Setup ---
    robot = None
    solver = None
    controller = None
    ik_worker = None
    state_container: dict[str, Any] = {
        "active": False,
        "solve_time": 0.0,
        "parse_time": 0.0,
        "xr_state": None,
    }

    camera_views = build_camera_views_config(
        head=cli.head_device,
        wrist_left=cli.wrist_left_device,
        wrist_right=cli.wrist_right_device,
        extra_streams=cli.camera,
    )

    robot_vis = None
    if cli.mode == "ik":
        logger.info("Initializing Unitree H1 robot and IK solver...")
        robot = UnitreeH1Robot()
        solver = PyrokiSolver(robot)
        controller = IKController(robot, solver)
        state_container["q"] = np.array(robot.get_default_config())
        robot_vis = robot.get_vis_config()

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        robot_vis=robot_vis,
        input_mode=cli.input_mode,
        camera_views=camera_views,
    )

    teleop = Teleop(settings=settings)
    teleop.set_pose(np.eye(4))

    # --- Event Processor Setup ---
    processor: Optional[EventProcessor] = None
    if cli.enable_events:
        event_settings = EventSettings(
            double_press_threshold_ms=cli.double_press_ms,
            long_press_threshold_ms=cli.long_press_ms,
        )
        processor = EventProcessor(event_settings)

        def log_event(event: ButtonEvent):
            event_log.append(event)
            # Log to general logs as well for headless/IK view
            logger.info(
                f"Event: {event.type.value} on {event.button.value} ({event.controller.value})"
            )

        processor.on_button_down(callback=log_event)
        processor.on_button_up(callback=log_event)
        processor.on_double_press(callback=log_event)
        processor.on_long_press(callback=log_event)

        # Robot Reset: Double-press on deadman switch (SQUEEZE)
        def on_reset_pose(event: ButtonEvent):
            if event.button == XRButton.SQUEEZE:
                # 1. Reset IK state if in IK mode
                # We check local variables robot, ik_worker, controller which are in scope
                # biome-ignore lint/style/noNonNullAssertion: ik_worker is checked
                if (
                    cli.mode == "ik"
                    and robot is not None
                    and ik_worker is not None
                    and controller is not None
                ):
                    # Thread-safe reset of configuration
                    default_q = np.array(robot.get_default_config())
                    state_container["q"] = default_q

                    # Reset controller engagement so it re-takes snapshots
                    controller.reset()

                    # Manually publish reset joint state so VR reflects it immediately
                    if ik_worker.teleop_loop:
                        joint_dict = {
                            name: float(val)
                            for name, val in zip(
                                robot.robot.joints.actuated_names, default_q
                            )
                        }
                        asyncio.run_coroutine_threadsafe(
                            teleop.publish_joint_state(joint_dict),
                            ik_worker.teleop_loop,
                        )

                    logger.info("Resetting Robot Joint State and IK Snapshots")
                else:
                    logger.info(
                        "Reset ignored: Not in IK mode or components not initialized"
                    )

        processor.on_double_press(button=XRButton.SQUEEZE, callback=on_reset_pose)

    # --- IK Worker Setup ---
    if cli.mode == "ik" and controller and robot:
        ik_worker = IKWorker(controller, robot, teleop, state_container, logger)
        ik_worker.start()

    # --- Teleop Callback ---
    def on_xr_update(_pose: np.ndarray, message: dict[str, Any]):
        try:
            # Capture loop for IK worker safety
            if ik_worker:
                try:
                    loop = asyncio.get_running_loop()
                    ik_worker.set_teleop_loop(loop)
                except RuntimeError:
                    pass

            t_parse_start = time.perf_counter()

            # Process events
            if processor:
                processor.process(_pose, message)

            xr_data = message.get("data", message)
            state = XRState.model_validate(xr_data)

            state_container["parse_time"] = time.perf_counter() - t_parse_start
            state_container["xr_state"] = state

            if ik_worker:
                ik_worker.update_state(state)
        except Exception:
            pass

    teleop.subscribe(on_xr_update)

    if cli.no_tui:
        print("TUI Disabled. Running in headless mode.")
        try:
            teleop.run()
        except KeyboardInterrupt:
            pass
        finally:
            if ik_worker:
                ik_worker.running = False
                ik_worker.join()
        return

    # --- TUI Loop ---
    console = Console()
    layout = Layout()

    if cli.mode == "ik":
        # Split: Left (State), Right (Top: IK Status, Middle: Controls, Bottom: Logs)
        layout.split_row(Layout(name="left", ratio=1), Layout(name="right", ratio=1))
        layout["right"].split_column(
            Layout(name="status", ratio=2),
            Layout(name="controls", size=5),
            Layout(name="logs", ratio=3),
        )
    else:
        # Split: Left (State), Right (Top: Events, Bottom: Legend)
        layout.split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
        layout["right"].split_column(
            Layout(name="events", ratio=4), Layout(name="help", size=3)
        )

    try:
        # Run Teleop in background thread
        t = threading.Thread(target=teleop.run, daemon=True)
        t.start()

        # Wait a bit for startup
        time.sleep(0.5)

        with Live(layout, refresh_per_second=10, console=console):
            while t.is_alive():
                # Common: Update State Table
                layout["left"].update(generate_state_table(state_container["xr_state"]))

                if cli.mode == "ik" and controller and robot:
                    layout["status"].update(
                        generate_ik_status_table(
                            state_container["active"],
                            state_container["solve_time"],
                            state_container["parse_time"],
                            state_container["xr_state"],
                            controller,
                            robot,
                            state_container.get("q", np.array([])),
                        )
                    )
                    layout["controls"].update(generate_ik_controls_panel())
                    layout["logs"].update(generate_log_panel(log_queue))
                else:
                    layout["events"].update(generate_event_panel(event_log))
                    layout["help"].update(generate_help_panel())

                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if ik_worker:
            ik_worker.running = False
            ik_worker.join()


if __name__ == "__main__":
    main()
