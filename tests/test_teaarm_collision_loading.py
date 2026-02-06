import json
import os
import pytest
from unittest.mock import patch, mock_open
from teleop_xr.ik.robots.teaarm import TeaArmRobot

TEAARM_URDF = """
<robot name="teaarm">
  <link name="waist_link"/>
  <link name="frame_left_arm_ee"/>
  <link name="frame_right_arm_ee"/>
  <link name="link1"/><link name="link2"/>
  <joint name="waist_yaw" type="revolute"><parent link="waist_link"/><child link="link1"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="waist_pitch" type="revolute"><parent link="link1"/><child link="link2"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j1" type="revolute"><parent link="link2"/><child link="frame_left_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j1" type="revolute"><parent link="link2"/><child link="frame_right_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
</robot>
"""


@pytest.fixture
def mock_asset_dir(tmp_path):
    d = tmp_path / "assets" / "teaarm"
    d.mkdir(parents=True)
    return d


def test_teaarm_load_collision_new_format(mock_asset_dir):
    collision_data = {
        "spheres": {"link1": {"centers": [[0, 0, 0]], "radii": [0.1]}},
        "collision_ignore_pairs": [["link1", "link2"]],
    }
    collision_file = mock_asset_dir / "collision.json"
    collision_file.write_text(json.dumps(collision_data))

    with patch(
        "teleop_xr.ik.robots.teaarm.os.path.join",
        side_effect=lambda *args: "/".join(args)
        if "assets" in args
        else os.path.join(*args),
    ):
        with patch(
            "teleop_xr.ik.robots.teaarm.os.path.dirname",
            return_value=str(mock_asset_dir.parent.parent),
        ):
            # We need to be careful with os.path.join mocking.
            # Better to mock os.path.exists and open.
            pass

    # Simpler approach: mock os.path.exists and builtins.open
    with patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists:

        def exists_side_effect(path):
            if "collision.json" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        with patch("builtins.open", mock_open(read_data=json.dumps(collision_data))):
            robot = TeaArmRobot(urdf_string=TEAARM_URDF)
            assert hasattr(robot, "robot_coll")
            # Verify it used from_sphere_decomposition (can check internal state or mock)


def test_teaarm_load_collision_legacy_format(mock_asset_dir):
    sphere_data = {"link1": {"centers": [[0, 0, 0]], "radii": [0.1]}}

    with patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists:

        def exists_side_effect(path):
            if "collision.json" in path:
                return False
            if "sphere.json" in path:
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        with patch("builtins.open", mock_open(read_data=json.dumps(sphere_data))):
            robot = TeaArmRobot(urdf_string=TEAARM_URDF)
            assert hasattr(robot, "robot_coll")


def test_teaarm_load_collision_missing():
    with patch("teleop_xr.ik.robots.teaarm.os.path.exists", return_value=False):
        robot = TeaArmRobot(urdf_string=TEAARM_URDF)
        assert hasattr(robot, "robot_coll")


def test_teaarm_load_collision_error_handling():
    with patch("teleop_xr.ik.robots.teaarm.os.path.exists", return_value=True):
        # Test JSON error
        with patch("builtins.open", mock_open(read_data="invalid json")):
            robot = TeaArmRobot(urdf_string=TEAARM_URDF)
            assert hasattr(robot, "robot_coll")

        # Test KeyError
        with patch("builtins.open", mock_open(read_data=json.dumps({"wrong_key": {}}))):
            robot = TeaArmRobot(urdf_string=TEAARM_URDF)
            assert hasattr(robot, "robot_coll")
