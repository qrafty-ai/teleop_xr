import numpy as np
from teleop import Teleop


def main():
    def callback(pose, xr_state):
        print(f"Pose: {pose}")
        print(f"XR State: {xr_state}")

    teleop = Teleop()
    teleop.set_pose(np.eye(4))
    teleop.subscribe(callback)
    teleop.run()


if __name__ == "__main__":
    main()
