import pytest

try:
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import jaxlie  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip(
        "jaxls, pyroki, jaxlie, or yourdfpy not installed", allow_module_level=True
    )

from unittest.mock import patch  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402

from teleop_xr.ik.robots.teaarm import TeaArmRobot  # noqa: E402


MINIMAL_TEAARM_URDF = """
<robot name="teaarm">
  <link name="base_link"/>
  <link name="waist_link"/>
  <joint name="waist_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="waist_link"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>

  <link name="torso_link"/>
  <joint name="waist_pitch" type="revolute">
    <parent link="waist_link"/>
    <child link="torso_link"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>

  <link name="left_arm_tip"/>
  <joint name="left_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="left_arm_tip"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>

  <link name="right_arm_tip"/>
  <joint name="right_joint" type="revolute">
    <parent link="torso_link"/>
    <child link="right_arm_tip"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>

  <link name="frame_left_arm_ee"/>
  <joint name="left_ee_joint" type="fixed">
    <parent link="left_arm_tip"/>
    <child link="frame_left_arm_ee"/>
  </joint>

  <link name="frame_right_arm_ee"/>
  <joint name="right_ee_joint" type="fixed">
    <parent link="right_arm_tip"/>
    <child link="frame_right_arm_ee"/>
  </joint>
</robot>
"""


def test_teaarm_init_with_string():
    robot = TeaArmRobot(urdf_string=MINIMAL_TEAARM_URDF)
    assert robot.L_ee == "frame_left_arm_ee"
    assert robot.R_ee == "frame_right_arm_ee"
    assert robot.supported_frames == {"left", "right"}
    assert len(robot.actuated_joint_names) == 4


def test_teaarm_forward_kinematics():
    robot = TeaArmRobot(urdf_string=MINIMAL_TEAARM_URDF)
    q = jnp.zeros(len(robot.actuated_joint_names))
    fk = robot.forward_kinematics(q)
    assert "left" in fk
    assert "right" in fk
    assert "head" in fk
    assert isinstance(fk["left"], jaxlie.SE3)
    assert isinstance(fk["right"], jaxlie.SE3)


def test_teaarm_build_costs():
    robot = TeaArmRobot(urdf_string=MINIMAL_TEAARM_URDF)
    target = jaxlie.SE3.identity()
    q = robot.get_default_config()
    costs = robot.build_costs(
        target_L=target,
        target_R=target,
        target_Head=None,
        q_current=q,
    )
    assert len(costs) >= 6


def test_teaarm_orientation_and_scale():
    robot = TeaArmRobot(urdf_string=MINIMAL_TEAARM_URDF)
    assert isinstance(robot.orientation, jaxlie.SO3)
    assert robot.model_scale == 1.0


def test_teaarm_get_vis_config_none():
    robot = TeaArmRobot(urdf_string=MINIMAL_TEAARM_URDF)
    assert robot.get_vis_config() is not None
    robot.urdf_path = ""
    assert robot.get_vis_config() is None


def test_teaarm_init_with_ram(tmp_path):
    dummy_urdf = tmp_path / "teaarm_asm_gen.urdf"
    dummy_urdf.write_text(MINIMAL_TEAARM_URDF)

    with patch("teleop_xr.ik.robots.teaarm.ram") as mock_ram:
        mock_ram.get_repo.return_value = tmp_path
        mock_ram.get_resource.return_value = dummy_urdf
        robot = TeaArmRobot()

        call_kwargs = mock_ram.get_resource.call_args[1]
        assert call_kwargs["xacro_args"]["visual_mesh_ext"] == "glb"
        assert mock_ram.get_repo.called
        assert robot.urdf_path == str(dummy_urdf)
        assert robot.mesh_path == str(tmp_path / "teaarm_description")


def test_teaarm_missing_left_ee():
    urdf = """
<robot name="teaarm">
  <link name="base_link"/>
  <link name="waist_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="waist_link"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>
  <link name="torso_link"/>
  <joint name="j2" type="fixed">
    <parent link="waist_link"/>
    <child link="torso_link"/>
  </joint>
  <link name="frame_right_arm_ee"/>
  <joint name="j3" type="fixed">
    <parent link="torso_link"/>
    <child link="frame_right_arm_ee"/>
  </joint>
</robot>
"""
    with pytest.raises(ValueError, match="Link frame_left_arm_ee not found"):
        TeaArmRobot(urdf_string=urdf)


def test_teaarm_missing_right_ee():
    urdf = """
<robot name="teaarm">
  <link name="base_link"/>
  <link name="waist_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="waist_link"/>
    <limit effort="1" lower="-1" upper="1" velocity="1"/>
  </joint>
  <link name="torso_link"/>
  <joint name="j2" type="fixed">
    <parent link="waist_link"/>
    <child link="torso_link"/>
  </joint>
  <link name="frame_left_arm_ee"/>
  <joint name="j3" type="fixed">
    <parent link="torso_link"/>
    <child link="frame_left_arm_ee"/>
  </joint>
</robot>
"""
    with pytest.raises(ValueError, match="Link frame_right_arm_ee not found"):
        TeaArmRobot(urdf_string=urdf)
