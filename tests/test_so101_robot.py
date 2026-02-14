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

from teleop_xr.ik.robots.so101 import SO101Robot  # noqa: E402

# Minimal SO101-like URDF with 5 arm joints + 1 gripper joint
MINIMAL_SO101_URDF = """
<robot name="so101">
  <link name="base_link"/>
  <link name="shoulder_link"/>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <limit effort="10" lower="-1.91986" upper="1.91986" velocity="10"/>
  </joint>
  <link name="upper_arm_link"/>
  <joint name="shoulder_lift" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <limit effort="10" lower="-1.74533" upper="1.74533" velocity="10"/>
  </joint>
  <link name="lower_arm_link"/>
  <joint name="elbow_flex" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="lower_arm_link"/>
    <limit effort="10" lower="-1.69" upper="1.69" velocity="10"/>
  </joint>
  <link name="wrist_link"/>
  <joint name="wrist_flex" type="revolute">
    <parent link="lower_arm_link"/>
    <child link="wrist_link"/>
    <limit effort="10" lower="-1.65806" upper="1.65806" velocity="10"/>
  </joint>
  <link name="gripper_link"/>
  <joint name="wrist_roll" type="revolute">
    <parent link="wrist_link"/>
    <child link="gripper_link"/>
    <limit effort="10" lower="-2.74385" upper="2.84121" velocity="10"/>
  </joint>
  <!-- Fixed gripper joint (simulates the fixed gripper in SO101) -->
  <link name="moving_jaw"/>
  <joint name="gripper" type="fixed">
    <parent link="gripper_link"/>
    <child link="moving_jaw"/>
  </joint>
</robot>
"""


@pytest.fixture
def mock_ram():
    with patch("teleop_xr.ik.robots.so101.ram") as mock_ram:
        yield mock_ram


@pytest.fixture
def dummy_urdf_file(tmp_path):
    p = tmp_path / "so101.urdf"
    p.write_text(MINIMAL_SO101_URDF)
    return p


def test_so101_init_with_string():
    """Test initialization with explicit URDF string."""
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    assert robot.ee_link_name == "gripper_link"
    assert robot.supported_frames == {"right"}
    # Should have 5 arm joints + 1 fixed gripper = 6 total
    assert (
        len(robot.actuated_joint_names) == 5
    )  # Only 5 revolute joints (gripper is fixed)


def test_so101_init_with_ram(mock_ram, dummy_urdf_file):
    """Test initialization using RAM (default)."""
    mock_ram.get_resource.return_value = dummy_urdf_file
    mock_ram.get_repo.return_value = dummy_urdf_file.parent

    robot = SO101Robot()

    # Check RAM calls
    assert mock_ram.get_resource.call_count == 1
    assert mock_ram.get_repo.called

    assert robot.urdf_path == str(dummy_urdf_file)
    assert robot.mesh_path == str(dummy_urdf_file.parent)


def test_so101_get_vis_config(mock_ram, dummy_urdf_file):
    mock_ram.get_resource.return_value = dummy_urdf_file
    mock_ram.get_repo.return_value = dummy_urdf_file.parent

    robot = SO101Robot()
    vis_config = robot.get_vis_config()

    assert vis_config is not None
    assert vis_config.urdf_path == str(dummy_urdf_file)
    assert vis_config.mesh_path == str(dummy_urdf_file.parent)
    assert vis_config.model_scale == 1.0


def test_so101_forward_kinematics():
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    q = jnp.zeros(5)
    fk = robot.forward_kinematics(q)

    assert "right" in fk
    assert isinstance(fk["right"], jaxlie.SE3)
    assert robot.joint_var_cls is not None


def test_so101_build_costs():
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    target = jaxlie.SE3.identity()

    costs = robot.build_costs(
        target_L=None, target_R=target, target_Head=None, q_current=jnp.zeros(5)
    )
    # Should have: rest cost, manipulability cost, pose cost, limit cost
    assert len(costs) >= 3

    costs_no_target = robot.build_costs(target_L=None, target_R=None, target_Head=None)
    # Should have: manipulability cost, limit cost (no rest or pose without target)
    assert len(costs_no_target) >= 2


def test_so101_get_vis_config_none():
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    assert robot.get_vis_config() is not None
    robot.urdf_path = ""
    assert robot.get_vis_config() is None


def test_so101_init_error(tmp_path):
    with (
        patch("teleop_xr.ik.robots.so101.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.so101.ram.get_repo") as mock_repo,
        patch("teleop_xr.ik.robots.so101.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = tmp_path / "nonexistent.urdf"
        mock_repo.return_value = tmp_path
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            SO101Robot()


def test_so101_default_config():
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    q = robot.get_default_config()
    # In minimal URDF we have 5 joints
    assert len(q) == 5
    assert jnp.allclose(q, jnp.zeros(5))


def test_so101_default_config_padding():
    """Test get_default_config with more joints than target (padding)."""
    # Create URDF with 8 joints
    joints_xml = ""
    for i in range(8):
        joints_xml += f'<joint name="j{i}" type="revolute"><parent link="l{i}"/><child link="l{i + 1}"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint><link name="l{i + 1}"/>'

    urdf = f'<robot name="many_joints"><link name="l0"/>{joints_xml}</robot>'

    robot = SO101Robot(urdf_string=urdf)
    q = robot.get_default_config()

    # Default config has 6 elements. Robot has 8.
    # Should pad with 2 zeros.
    assert len(q) == 8
    assert q[6] == 0.0
    assert q[7] == 0.0


def test_so101_orientation():
    """Test that orientation property returns identity SO3."""
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    ori = robot.orientation
    assert isinstance(ori, jaxlie.SO3)
    # Identity rotation
    assert jnp.allclose(ori.as_matrix(), jnp.eye(3))


def test_so101_model_scale():
    """Test model scale property."""
    robot = SO101Robot(urdf_string=MINIMAL_SO101_URDF)
    assert robot.model_scale == 1.0
