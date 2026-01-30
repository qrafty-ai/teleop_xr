"""TeleopXR Demo - Rich TUI for XR state and event visualization.

This demo showcases both real-time XR device tracking and the event system
for detecting button gestures (press, release, double-press, long-press).
"""

import numpy as np
import tyro
from collections import deque
from dataclasses import dataclass
from typing import Union, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.common_cli import CommonCLI
from teleop_xr.messages import XRState
from teleop_xr.camera_views import build_camera_views_config
from teleop_xr.events import (
    EventProcessor,
    EventSettings,
    ButtonEvent,
    ButtonEventType,
)


# Maximum number of events to display in the event log
MAX_EVENT_LOG_SIZE = 10


@dataclass
class DemoCLI(CommonCLI):
    """CLI options for the TeleopXR demo."""

    # Camera device configuration
    head_device: Union[int, str, None] = None
    """Camera device for head view (index or path)."""
    wrist_left_device: Union[int, str, None] = None
    """Camera device for left wrist view (index or path)."""
    wrist_right_device: Union[int, str, None] = None
    """Camera device for right wrist view (index or path)."""

    # Event system configuration
    double_press_ms: float = 300
    """Maximum time between presses to count as double-press (ms)."""
    long_press_ms: float = 500
    """Minimum hold time to count as long-press (ms)."""
    enable_events: bool = True
    """Enable the event system for gesture detection."""


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
        # WebXR sends controllers as gripPose (or pose), check both
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

            # Show pressed buttons indices
            pressed = [i for i, b in enumerate(buttons) if b.pressed]
            if pressed:
                inputs_parts.append(f"Btn:{pressed}")

            # Show non-zero axes
            active_axes = [f"{i}:{v:.1f}" for i, v in enumerate(axes) if abs(v) > 0.1]
            if active_axes:
                inputs_parts.append(f"Ax:{','.join(active_axes)}")

        # Joints count if present
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

            # Build the event line
            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(f"[{controller}] ", style="bold white")
            line.append(f"{button} ", style="cyan")
            line.append(f"{event_name}", style=style)

            # Add hold duration for button up events
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


def generate_layout(
    xr_state: Optional[XRState],
    event_log: deque,
    events_enabled: bool,
) -> Layout:
    """Generate the full TUI layout."""
    layout = Layout()

    if events_enabled:
        # Two-column layout: state table (left) and events (right)
        layout.split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )

        layout["left"].update(generate_state_table(xr_state))

        # Stack event panel and help on the right
        layout["right"].split_column(
            Layout(generate_event_panel(event_log), name="events", ratio=4),
            Layout(generate_help_panel(), name="help", size=3),
        )
    else:
        # Single-column layout: just the state table
        layout.update(generate_state_table(xr_state))

    return layout


def main():
    cli = tyro.cli(DemoCLI)

    # Backward compatibility: default to head on device 0 if no flags provided
    if (
        cli.head_device is None
        and cli.wrist_left_device is None
        and cli.wrist_right_device is None
    ):
        cli.head_device = 0  # Default to index 0

    camera_views = build_camera_views_config(
        head=cli.head_device,
        wrist_left=cli.wrist_left_device,
        wrist_right=cli.wrist_right_device,
    )

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        input_mode=cli.input_mode,
        camera_views=camera_views,
    )

    teleop = Teleop(settings=settings)
    teleop.set_pose(np.eye(4))

    # Event log (thread-safe deque with max size)
    event_log: deque[ButtonEvent] = deque(maxlen=MAX_EVENT_LOG_SIZE)

    # Event processor setup
    processor: Optional[EventProcessor] = None
    if cli.enable_events:
        event_settings = EventSettings(
            double_press_threshold_ms=cli.double_press_ms,
            long_press_threshold_ms=cli.long_press_ms,
        )
        processor = EventProcessor(event_settings)

        # Register callbacks that add events to the log
        def log_event(event: ButtonEvent):
            event_log.append(event)

        processor.on_button_down(callback=log_event)
        processor.on_button_up(callback=log_event)
        processor.on_double_press(callback=log_event)
        processor.on_long_press(callback=log_event)

    console = Console()

    initial_layout = generate_layout(None, event_log, cli.enable_events)

    with Live(initial_layout, refresh_per_second=20, console=console) as live:

        def callback(pose, xr_state_dict):
            try:
                # Process events first (if enabled)
                if processor:
                    processor.process(pose, xr_state_dict)

                # Validate the incoming dict against the Pydantic model
                xr_state = XRState.model_validate(xr_state_dict)

                # Update the display
                live.update(generate_layout(xr_state, event_log, cli.enable_events))
            except Exception:
                # In case of validation error or other issues, just ignore to keep UI running
                pass

        teleop.subscribe(callback)

        # Build startup information message
        startup_lines = [
            "[bold]TeleopXR Demo[/bold]\n",
            f"• Server: https://{cli.host}:{cli.port}",
            f"• Event detection: {'[green]enabled[/green]' if cli.enable_events else '[yellow]disabled[/yellow]'}",
        ]
        if cli.enable_events:
            startup_lines.append(f"• Double-press threshold: {cli.double_press_ms}ms")
            startup_lines.append(f"• Long-press threshold: {cli.long_press_ms}ms")
        startup_lines.append("\n[dim]Press Ctrl+C to exit[/dim]")

        # Print startup information
        console.print(
            Panel(
                "\n".join(startup_lines),
                title="[bold cyan]Starting...[/bold cyan]",
                box=box.DOUBLE,
            )
        )

        try:
            teleop.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")


if __name__ == "__main__":
    main()
