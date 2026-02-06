import io
import pytest
import yourdfpy
from ballpark import Robot
from scripts.configure_sphere_collision import (
    compute_collision_ignore_pairs,
    _process_chunk_worker,
)


@pytest.fixture
def mock_robot():
    urdf_str = """
    <robot name="test">
      <link name="l1"><collision><origin xyz="0 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
      <link name="l2"><collision><origin xyz="0.2 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
      <link name="l3"><collision><origin xyz="0.4 0 0"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
      <joint name="j1" type="fixed"><parent link="l1"/><child link="l2"/><origin xyz="0.2 0 0"/></joint>
      <joint name="j2" type="fixed"><parent link="l2"/><child link="l3"/><origin xyz="0.2 0 0"/></joint>
    </robot>
    """
    urdf = yourdfpy.URDF.load(io.StringIO(urdf_str))
    return Robot(urdf)


def test_compute_collision_ignore_pairs_serial(mock_robot):
    # l1 and l3 are non-contiguous and never collide
    ignore_pairs = compute_collision_ignore_pairs(mock_robot, n_samples=5, n_jobs=1)

    # Check if ("l1", "l3") is in ignore pairs
    found = {tuple(sorted(p)) for p in ignore_pairs}
    assert ("l1", "l3") in found


def test_compute_collision_ignore_pairs_parallel(mock_robot):
    # Test parallel branch
    ignore_pairs = compute_collision_ignore_pairs(mock_robot, n_samples=4, n_jobs=2)
    assert len(ignore_pairs) > 0


def test_worker_function(mock_robot):
    lower, upper = mock_robot.joint_limits
    non_contiguous = mock_robot.non_contiguous_pairs

    counts = _process_chunk_worker(
        mock_robot,
        count=2,
        lower=lower,
        upper=upper,
        threshold=0.01,
        non_contiguous=non_contiguous,
    )

    assert isinstance(counts, dict)
    assert len(counts) == len(non_contiguous)


def test_compute_collision_no_pairs():
    # Robot with no non-contiguous pairs
    urdf_str = """<robot name="t"><link name="l1"/></robot>"""
    urdf = yourdfpy.URDF.load(io.StringIO(urdf_str))
    robot = Robot(urdf)
    assert compute_collision_ignore_pairs(robot) == []
