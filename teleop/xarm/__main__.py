import numpy as np
from teleop import Teleop
import time
import transforms3d as t3d

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    raise ImportError(
        "xarm-python-sdk is not installed. Please install the dependency with `pip install xarm-python-sdk`."
    )


class Lite6Gripper:
    def __init__(self, arm: XArmAPI):
        self._arm = arm
        self._prev_gripper_state = None
        self._gripper_state = 0.0
        self._gripper_open_time = 0.0
        self._gripper_stopped = False

    def open(self):
        self._arm.open_lite6_gripper()

    def close(self):
        self._arm.close_lite6_gripper()

    def stop(self):
        self._arm.stop_lite6_gripper()

    def set_gripper_state(self, gripper_state: float) -> None:
        """Set gripper state and handle opening/closing logic."""
        self._gripper_state = gripper_state

        if self._gripper_state is not None:
            if (
                self._prev_gripper_state is None
                or self._prev_gripper_state != self._gripper_state
            ):
                if self._gripper_state < 1.0:
                    self._gripper_stopped = False
                    self.close()
                else:
                    self._gripper_open_time = time.time()
                    self._gripper_stopped = False
                    self.open()

            if (
                not self._gripper_stopped
                and self._gripper_state >= 1.0
                and time.time() - self._gripper_open_time > 1.0
            ):
                # If gripper was closed and now is open, stop the gripper
                self._gripper_stopped = True
                self.stop()

        self._prev_gripper_state = self._gripper_state

    def get_gripper_state(self) -> float:
        """Get current gripper state."""
        return self._gripper_state

    def reset_gripper(self) -> None:
        """Reset gripper state."""
        self._prev_gripper_state = None
        self._gripper_state = 0.0
        self._gripper_open_time = 0.0
        self._gripper_stopped = False


def get_pose(arm):
    ok, pose = arm.get_position()
    if ok != 0:
        raise RuntimeError(f"Failed to get arm position: {ok}")

    translation = np.array(pose[:3]) / 1000
    eulers = np.array(pose[3:])
    rotation = t3d.euler.euler2mat(eulers[0], eulers[1], eulers[2], "sxyz")
    pose = t3d.affines.compose(translation, rotation, np.ones(3))
    return pose


def servo(arm, pose):
    x = pose[0, 3] * 1000
    y = pose[1, 3] * 1000
    z = pose[2, 3] * 1000
    roll, pitch, yaw = t3d.euler.mat2euler(pose[:3, :3])
    error = arm.set_servo_cartesian([x, y, z, roll, pitch, yaw], speed=100, mvacc=100)
    return error


def main():
    arm = XArmAPI("192.168.1.184", is_radian=True)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(state=0)

    gripper = Lite6Gripper(arm)

    def callback(pose, xr_state):
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

        if move:
            servo(arm, pose)

        # Map gripper to controller grip: buttons[1] value/pressed; output string "close" if grip > 0.5 else "open"
        gripper_command = "open"
        if len(buttons) > 1:
            grip_val = buttons[1].get(
                "value", 1.0 if buttons[1].get("pressed", False) else 0.0
            )
            gripper_command = "close" if grip_val > 0.5 else "open"

        gripper_state = 1.0 if gripper_command == "close" else 0.0
        gripper.set_gripper_state(gripper_state)

    teleop = Teleop(natural_phone_orientation_euler=[0, 0, 0])
    teleop.set_pose(get_pose(arm))
    teleop.subscribe(callback)
    teleop.run()


if __name__ == "__main__":
    main()
