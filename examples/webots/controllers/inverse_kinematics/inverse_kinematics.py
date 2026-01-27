import tempfile
import threading
from controller import Robot
from teleop import Teleop
from teleop.utils.jacobi_robot import JacobiRobot


class RobotArm(Robot):
    def __init__(self):
        super().__init__()
        self.__motors = {}

        self.__jacobi = None
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as file:
            file.write(self.getUrdf().encode("utf-8"))
            self.__jacobi = JacobiRobot(file.name, ee_link="wrist_3_link")

        # Initialize the arm motors and encoders.
        self.timestep = int(self.getBasicTimeStep())
        for joint_name in self.__jacobi.get_joint_names():
            motor = self.getDevice(joint_name)
            motor.setVelocity(1.0)
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(self.timestep)
            self.__motors[joint_name] = motor

    def move_to_position(self, pose):
        self.__jacobi.servo_to_pose(pose, dt=self.timestep / 1000.0)
        for name, motor in self.__motors.items():
            motor.setPosition(self.__jacobi.get_joint_position(name))

    def move_joints(self, positions):
        for name, motor in self.__motors.items():
            motor.setPosition(positions[name])
            self.__jacobi.set_joint_position(name, positions[name])

    def get_current_pose(self):
        return self.__jacobi.get_ee_pose()


def main():
    robot = RobotArm()
    teleop = Teleop(natural_phone_orientation_euler=[0, 0, 0])
    target_pose = None

    def on_teleop_callback(pose, xr_state):
        nonlocal target_pose

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
            target_pose = None
            return

        buttons = controller.get("gamepad", {}).get("buttons", [])
        # Map move gating to controller trigger: buttons[0].pressed (or value > 0.5 if present)
        move = False
        if len(buttons) > 0:
            move = (
                buttons[0].get("pressed", False) or buttons[0].get("value", 0.0) > 0.5
            )

        if move:
            target_pose = pose
        else:
            target_pose = None

    robot.move_joints(
        {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.0,
            "elbow_joint": 1.8,
            "wrist_1_joint": -2.0,
            "wrist_2_joint": -1.3,
            "wrist_3_joint": 0.4,
        }
    )
    timestep = int(robot.getBasicTimeStep())
    for _ in range(100):
        robot.step(timestep) != -1

    current_pose = robot.get_current_pose()
    teleop.set_pose(current_pose)

    teleop.subscribe(on_teleop_callback)
    thread = threading.Thread(target=teleop.run)
    thread.start()

    while robot.step(timestep) != -1:
        if target_pose is not None:
            robot.move_to_position(target_pose)

    teleop.stop()
    thread.join()


if __name__ == "__main__":
    main()
