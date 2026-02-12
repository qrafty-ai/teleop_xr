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
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402

from teleop_xr.ik.robots.openarm import OpenArmRobot  # noqa: E402

# Minimal bimanual URDF matching the link/joint naming convention of openarm
MINIMAL_OPENARM_URDF = """
<robot name="openarm">
  <link name="world"/>
  <link name="openarm_body_link0"/>
  <joint name="openarm_body_world_joint" type="fixed">
    <parent link="world"/>
    <child link="openarm_body_link0"/>
  </joint>

  <!-- Left arm -->
  <link name="openarm_left_link0"/>
  <joint name="openarm_left_openarm_body_link0_joint" type="fixed">
    <parent link="openarm_body_link0"/>
    <child link="openarm_left_link0"/>
  </joint>
  <link name="openarm_left_link1"/>
  <joint name="openarm_left_joint1" type="revolute">
    <parent link="openarm_left_link0"/>
    <child link="openarm_left_link1"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link2"/>
  <joint name="openarm_left_joint2" type="revolute">
    <parent link="openarm_left_link1"/>
    <child link="openarm_left_link2"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link3"/>
  <joint name="openarm_left_joint3" type="revolute">
    <parent link="openarm_left_link2"/>
    <child link="openarm_left_link3"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link4"/>
  <joint name="openarm_left_joint4" type="revolute">
    <parent link="openarm_left_link3"/>
    <child link="openarm_left_link4"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link5"/>
  <joint name="openarm_left_joint5" type="revolute">
    <parent link="openarm_left_link4"/>
    <child link="openarm_left_link5"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link6"/>
  <joint name="openarm_left_joint6" type="revolute">
    <parent link="openarm_left_link5"/>
    <child link="openarm_left_link6"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_left_link7"/>
  <joint name="openarm_left_joint7" type="revolute">
    <parent link="openarm_left_link6"/>
    <child link="openarm_left_link7"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>

  <!-- Right arm -->
  <link name="openarm_right_link0"/>
  <joint name="openarm_right_openarm_body_link0_joint" type="fixed">
    <parent link="openarm_body_link0"/>
    <child link="openarm_right_link0"/>
  </joint>
  <link name="openarm_right_link1"/>
  <joint name="openarm_right_joint1" type="revolute">
    <parent link="openarm_right_link0"/>
    <child link="openarm_right_link1"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link2"/>
  <joint name="openarm_right_joint2" type="revolute">
    <parent link="openarm_right_link1"/>
    <child link="openarm_right_link2"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link3"/>
  <joint name="openarm_right_joint3" type="revolute">
    <parent link="openarm_right_link2"/>
    <child link="openarm_right_link3"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link4"/>
  <joint name="openarm_right_joint4" type="revolute">
    <parent link="openarm_right_link3"/>
    <child link="openarm_right_link4"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link5"/>
  <joint name="openarm_right_joint5" type="revolute">
    <parent link="openarm_right_link4"/>
    <child link="openarm_right_link5"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link6"/>
  <joint name="openarm_right_joint6" type="revolute">
    <parent link="openarm_right_link5"/>
    <child link="openarm_right_link6"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
  <link name="openarm_right_link7"/>
  <joint name="openarm_right_joint7" type="revolute">
    <parent link="openarm_right_link6"/>
    <child link="openarm_right_link7"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.175"/>
  </joint>
</robot>
"""


def test_openarm_init_with_string():
    """Test initialization with explicit URDF string."""
    robot = OpenArmRobot(urdf_string=MINIMAL_OPENARM_URDF)
    assert robot.L_ee == "openarm_left_link7"
    assert robot.R_ee == "openarm_right_link7"
    assert robot.supported_frames == {"left", "right"}
    assert len(robot.actuated_joint_names) == 14  # 7 per arm


def test_openarm_forward_kinematics():
    """Test forward kinematics returns expected frames."""
    robot = OpenArmRobot(urdf_string=MINIMAL_OPENARM_URDF)
    q = jnp.zeros(len(robot.actuated_joint_names))
    fk = robot.forward_kinematics(q)

    assert "left" in fk
    assert "right" in fk
    assert isinstance(fk["left"], jaxlie.SE3)
    assert isinstance(fk["right"], jaxlie.SE3)
    assert robot.joint_var_cls is not None


def test_openarm_build_costs():
    """Test cost building with all targets."""
    robot = OpenArmRobot(urdf_string=MINIMAL_OPENARM_URDF)
    target = jaxlie.SE3.identity()
    q = robot.get_default_config()

    costs = robot.build_costs(
        target_L=target, target_R=target, target_Head=None, q_current=q
    )
    # rest + manipulability + L + R + limits + self_collision = 6
    assert len(costs) == 6

    costs_no_target = robot.build_costs(target_L=None, target_R=None, target_Head=None)
    # manipulability + limits + self_collision = 3
    assert len(costs_no_target) == 3


def test_openarm_get_vis_config_none():
    """Test vis config is None when initialized with string."""
    robot = OpenArmRobot(urdf_string=MINIMAL_OPENARM_URDF)
    assert robot.get_vis_config() is None


def test_openarm_default_config():
    """Test default config returns correct length."""
    robot = OpenArmRobot(urdf_string=MINIMAL_OPENARM_URDF)
    q = robot.get_default_config()
    assert len(q) == 14
    assert isinstance(q, jax.Array)


def test_openarm_init_with_ram(tmp_path):
    """Test initialization using RAM (default)."""
    dummy_urdf = tmp_path / "v10.urdf"
    dummy_urdf.write_text(MINIMAL_OPENARM_URDF)

    with (
        patch("teleop_xr.ik.robots.openarm.ram") as mock_ram,
        patch("teleop_xr.ik.robots.openarm.os.path.exists") as mock_exists,
    ):
        mock_ram.get_resource.return_value = dummy_urdf
        mock_ram.get_repo.return_value = tmp_path
        mock_exists.return_value = True

        robot = OpenArmRobot()

        assert mock_ram.get_resource.call_count == 1
        assert mock_ram.get_repo.called

        # Verify xacro args include bimanual=true
        call_kwargs = mock_ram.get_resource.call_args[1]
        assert call_kwargs["xacro_args"]["bimanual"] == "true"

        assert robot.urdf_path == str(dummy_urdf)
        assert robot.mesh_path == str(tmp_path)
        assert robot.get_vis_config() is not None


def test_openarm_init_error(tmp_path):
    """Test FileNotFoundError when URDF doesn't exist."""
    with (
        patch("teleop_xr.ik.robots.openarm.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.openarm.ram.get_repo") as mock_repo,
        patch("teleop_xr.ik.robots.openarm.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = tmp_path / "nonexistent.urdf"
        mock_repo.return_value = tmp_path
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="OpenArm URDF not found"):
            OpenArmRobot()


def test_openarm_missing_left_ee():
    """Test error when left EE link is missing."""
    urdf = """<robot name="m">
    <link name="openarm_body_link0"/>
    <link name="openarm_right_link7"/>
    <joint name="j1" type="revolute">
      <parent link="openarm_body_link0"/>
      <child link="openarm_right_link7"/>
      <limit effort="1" lower="-1" upper="1" velocity="1"/>
    </joint>
    </robot>"""
    with pytest.raises(ValueError, match="Link openarm_left_link7 not found"):
        OpenArmRobot(urdf_string=urdf)


def test_openarm_missing_right_ee():
    """Test error when right EE link is missing."""
    urdf = """<robot name="m">
    <link name="openarm_body_link0"/>
    <link name="openarm_left_link7"/>
    <joint name="j1" type="revolute">
      <parent link="openarm_body_link0"/>
      <child link="openarm_left_link7"/>
      <limit effort="1" lower="-1" upper="1" velocity="1"/>
    </joint>
    </robot>"""
    with pytest.raises(ValueError, match="Link openarm_right_link7 not found"):
        OpenArmRobot(urdf_string=urdf)
