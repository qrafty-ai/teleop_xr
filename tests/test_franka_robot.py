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

# Mock dependencies before importing FrankaRobot if they do heavy lifting at import time
# FrankaRobot imports pyroki, yourdfpy, etc.

from teleop_xr.ik.robots.franka import FrankaRobot  # noqa: E402

MINIMAL_FRANKA_URDF = """
<robot name="panda">
  <link name="panda_link0"/>
  <link name="panda_link1"/>
  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <limit effort="87" lower="-2.8973" upper="2.8973" velocity="2.1750"/>
  </joint>
  <!-- simplified chain -->
  <link name="panda_hand"/>
  <joint name="panda_hand_joint" type="fixed">
    <parent link="panda_link1"/>
    <child link="panda_hand"/>
  </joint>
</robot>
"""


@pytest.fixture
def mock_ram():
    with patch("teleop_xr.ik.robots.franka.ram") as mock_ram:
        # Mock get_resource to return a path to a dummy URDF
        # We need to write the URDF to a file because FrankaRobot loads it via filename if not string
        # Actually FrankaRobot can take urdf_string in __init__, but the default path uses RAM.

        # Let's mock get_resource to return a Path object that points to our dummy URDF
        # However, FrankaRobot init calls ram.get_resource(...), converts to str, then checks os.path.exists

        yield mock_ram


@pytest.fixture
def dummy_urdf_file(tmp_path):
    p = tmp_path / "panda.urdf"
    p.write_text(MINIMAL_FRANKA_URDF)
    return p


def test_franka_init_with_string():
    """Test initialization with explicit URDF string."""
    robot = FrankaRobot(urdf_string=MINIMAL_FRANKA_URDF)
    assert robot.ee_link_name == "panda_hand"
    assert robot.supported_frames == {"right"}
    assert len(robot.actuated_joint_names) == 1  # Only 1 revolute joint in minimal URDF


def test_franka_init_with_ram(mock_ram, dummy_urdf_file):
    """Test initialization using RAM (default)."""
    # Setup mock to return the dummy file path
    mock_ram.get_resource.return_value = dummy_urdf_file
    mock_ram.get_repo.return_value = dummy_urdf_file.parent

    robot = FrankaRobot()

    # Check RAM calls
    assert mock_ram.get_resource.call_count == 1  # Once for IK (and Vis uses same)
    assert mock_ram.get_repo.called

    assert robot.urdf_path == str(dummy_urdf_file)
    # assert robot.vis_urdf_path == str(dummy_urdf_file) # Removed property
    assert robot.mesh_path == str(dummy_urdf_file.parent)


def test_franka_get_vis_config(mock_ram, dummy_urdf_file):
    mock_ram.get_resource.return_value = dummy_urdf_file
    mock_ram.get_repo.return_value = dummy_urdf_file.parent

    robot = FrankaRobot()
    vis_config = robot.get_vis_config()

    assert vis_config is not None
    assert vis_config.urdf_path == str(dummy_urdf_file)
    assert vis_config.mesh_path == str(dummy_urdf_file.parent)
    assert vis_config.model_scale == 0.5


def test_franka_forward_kinematics():
    robot = FrankaRobot(urdf_string=MINIMAL_FRANKA_URDF)
    q = jnp.zeros(1)
    fk = robot.forward_kinematics(q)

    assert "right" in fk
    assert isinstance(fk["right"], jaxlie.SE3)
    assert robot.joint_var_cls is not None


def test_franka_build_costs():
    robot = FrankaRobot(urdf_string=MINIMAL_FRANKA_URDF)
    target = jaxlie.SE3.identity()

    costs = robot.build_costs(
        target_L=None, target_R=target, target_Head=None, q_current=jnp.zeros(1)
    )
    assert len(costs) >= 3

    costs_no_target = robot.build_costs(target_L=None, target_R=None, target_Head=None)
    assert len(costs_no_target) == 2


def test_franka_get_vis_config_none():
    robot = FrankaRobot(urdf_string=MINIMAL_FRANKA_URDF)
    assert robot.get_vis_config() is None


def test_franka_init_error(tmp_path):
    with (
        patch("teleop_xr.ik.robots.franka.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.franka.ram.get_repo") as mock_repo,
        patch("teleop_xr.ik.robots.franka.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = tmp_path / "nonexistent.urdf"
        mock_repo.return_value = tmp_path
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            FrankaRobot()


def test_franka_default_config():
    robot = FrankaRobot(urdf_string=MINIMAL_FRANKA_URDF)
    q = robot.get_default_config()
    # In minimal URDF we have 1 joint.
    # get_default_config hardcodes a 7-DOF pose and pads/truncates.
    # q_target has 7 elements.
    # Our robot has 1 joint. q should be truncated to 1.
    assert len(q) == 1
    assert q[0] == 0.0  # First element of q_target is 0.0


def test_franka_default_config_padding():
    """Test get_default_config with more joints than target (padding)."""
    # Create URDF with 9 joints
    joints_xml = ""
    for i in range(9):
        joints_xml += f'<joint name="j{i}" type="revolute"><parent link="l{i}"/><child link="l{i + 1}"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint><link name="l{i + 1}"/>'

    urdf = f'<robot name="many_joints"><link name="l0"/>{joints_xml}</robot>'

    robot = FrankaRobot(urdf_string=urdf)
    q = robot.get_default_config()

    # Target has 7 elements. Robot has 9.
    # Should pad with 2 zeros.
    assert len(q) == 9
    assert q[7] == 0.0
    assert q[8] == 0.0
