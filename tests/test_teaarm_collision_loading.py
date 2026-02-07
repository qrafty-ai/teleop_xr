import json
import os
import tempfile
from unittest.mock import mock_open, patch

import pytest

from teleop_xr.ik.robots.teaarm import TeaArmRobot

_TMPDIR = tempfile.mkdtemp()


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


def _make_teaarm_with_collision_mock(urdf_text, exists_side_effect, open_mock=None):
    """Create TeaArmRobot with mocked RAM + collision file mocks.

    Writes ``TEAARM_URDF`` to a real temp file so that ``yourdfpy.URDF.load``
    succeeds during ``__init__``, then uses ``set_description()`` with the
    collision mocks active for the reinitialisation.
    """
    urdf_file = os.path.join(_TMPDIR, "teaarm.urdf")
    with open(urdf_file, "w") as f:
        f.write(TEAARM_URDF)

    with patch("teleop_xr.ik.robots.teaarm.ram.get_resource") as mock_get:
        mock_get.return_value = urdf_file
        robot = TeaArmRobot()

    with patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists:
        mock_exists.side_effect = exists_side_effect
        if open_mock is not None:
            with open_mock:
                robot.set_description(urdf_text)
        else:
            robot.set_description(urdf_text)
    return robot


def test_teaarm_load_collision_new_format(mock_asset_dir):
    collision_data = {
        "spheres": {"link1": {"centers": [[0, 0, 0]], "radii": [0.1]}},
        "collision_ignore_pairs": [["link1", "link2"]],
    }

    def exists_side_effect(path):
        return "collision.json" in str(path)

    robot = _make_teaarm_with_collision_mock(
        TEAARM_URDF,
        exists_side_effect,
        patch("builtins.open", mock_open(read_data=json.dumps(collision_data))),
    )
    assert hasattr(robot, "robot_coll")


def test_teaarm_load_collision_legacy_format(mock_asset_dir):
    sphere_data = {"link1": {"centers": [[0, 0, 0]], "radii": [0.1]}}

    def exists_side_effect(path):
        return "sphere.json" in str(path)

    robot = _make_teaarm_with_collision_mock(
        TEAARM_URDF,
        exists_side_effect,
        patch("builtins.open", mock_open(read_data=json.dumps(sphere_data))),
    )
    assert hasattr(robot, "robot_coll")


def test_teaarm_load_collision_missing():
    robot = _make_teaarm_with_collision_mock(TEAARM_URDF, lambda _: False)
    assert hasattr(robot, "robot_coll")


def test_teaarm_load_collision_error_handling():
    def exists_side_effect(path):
        return True

    # Test JSON error
    robot = _make_teaarm_with_collision_mock(
        TEAARM_URDF,
        exists_side_effect,
        patch("builtins.open", mock_open(read_data="invalid json")),
    )
    assert hasattr(robot, "robot_coll")

    # Test KeyError
    robot2 = _make_teaarm_with_collision_mock(
        TEAARM_URDF,
        exists_side_effect,
        patch("builtins.open", mock_open(read_data=json.dumps({"wrong_key": {}}))),
    )
    assert hasattr(robot2, "robot_coll")
