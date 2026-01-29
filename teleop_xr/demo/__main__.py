import numpy as np
import tyro
from dataclasses import dataclass
from typing import Union, Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.common_cli import CommonCLI
from teleop_xr.messages import XRState
from teleop_xr.camera_views import build_camera_views_config


@dataclass
class DemoCLI(CommonCLI):
    # Devices can be int (index) or str (path)
    head_device: Union[int, str, None] = None
    wrist_left_device: Union[int, str, None] = None
    wrist_right_device: Union[int, str, None] = None


def generate_table(xr_state: Optional[XRState] = None) -> Table:
    table = Table(title="WebXR Teleop State", box=box.ROUNDED, expand=True)
    table.add_column("Role", style="cyan", no_wrap=True)
    table.add_column("Hand", style="magenta")
    table.add_column("Position (x, y, z)", style="green")
    table.add_column("Orientation (x, y, z, w)", style="yellow")
    table.add_column("Inputs", style="blue")

    if not xr_state or not xr_state.devices:
        table.add_row("-", "-", "-", "-", "Waiting for data...")
        return table

    devices = xr_state.devices

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

    console = Console()

    with Live(generate_table(), refresh_per_second=20, console=console) as live:

        def callback(pose, xr_state_dict):
            try:
                # Validate the incoming dict against the Pydantic model
                xr_state = XRState.model_validate(xr_state_dict)
                live.update(generate_table(xr_state))
            except Exception:
                # In case of validation error or other issues, just ignore to keep UI running
                pass

        teleop.subscribe(callback)

        try:
            teleop.run()
        except KeyboardInterrupt:
            pass

        teleop.subscribe(callback)

        try:
            teleop.run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
