import argparse
import numpy as np
from teleop import Teleop
from teleop.camera_views import (
    build_camera_views_config,
    build_video_streams,
    parse_device_spec,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-device")
    parser.add_argument("--wrist-left-device")
    parser.add_argument("--wrist-right-device")
    args = parser.parse_args()

    try:
        head = (
            parse_device_spec(args.head_device)
            if args.head_device is not None
            else None
        )
        wrist_left = (
            parse_device_spec(args.wrist_left_device)
            if args.wrist_left_device is not None
            else None
        )
        wrist_right = (
            parse_device_spec(args.wrist_right_device)
            if args.wrist_right_device is not None
            else None
        )

        # Backward compatibility: default to head on device 0 if no flags provided
        if head is None and wrist_left is None and wrist_right is None:
            head = 0

        camera_views = build_camera_views_config(
            head=head,
            wrist_left=wrist_left,
            wrist_right=wrist_right,
        )
    except ValueError as e:
        parser.error(str(e))

    def callback(pose, xr_state):
        return
        print(f"Pose: {pose}")
        print(f"XR State: {xr_state}")

    teleop = Teleop(camera_views=camera_views)
    teleop.set_pose(np.eye(4))
    teleop.set_video_streams(build_video_streams(camera_views))
    teleop.subscribe(callback)
    teleop.run()


if __name__ == "__main__":
    main()
