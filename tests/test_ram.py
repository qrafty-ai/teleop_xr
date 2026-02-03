import tempfile
from pathlib import Path

import pytest
import git
from teleop_xr import ram


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Fixture to provide a temporary cache directory for RAM."""
    cache_dir = tmp_path / "ram_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_git_repo():
    """Fixture to create a local git repository with dummy assets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / "test_repo"
        repo_path.mkdir()

        # Create dummy assets
        urdf_path = repo_path / "robot.urdf"
        urdf_path.write_text(
            '<robot name="test_robot"><link name="base_link"/></robot>'
        )

        xacro_path = repo_path / "robot.xacro"
        xacro_path.write_text(
            '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="test_xacro"><link name="base"/></robot>'
        )

        mesh_dir = repo_path / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "base.stl").write_text("dummy mesh content")

        # Initialize git repo
        repo = git.Repo.init(repo_path)
        repo.index.add(["robot.urdf", "robot.xacro", "meshes/base.stl"])
        repo.index.commit("Initial commit")

        yield repo_path


def test_get_repo(temp_git_repo, mock_cache_dir):
    """Test get_repo function."""
    repo_url = f"file://{temp_git_repo}"
    repo_dir = ram.get_repo(repo_url, cache_dir=mock_cache_dir)
    assert repo_dir.exists()
    assert (repo_dir / "robot.urdf").exists()


def test_fetch_git_repo(temp_git_repo, mock_cache_dir):
    """Test that RAM can fetch a git repo into cache via get_resource."""
    repo_url = f"file://{temp_git_repo}"
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot.urdf", cache_dir=mock_cache_dir
    )
    assert asset_path.exists()
    assert asset_path.suffix == ".urdf"
    assert asset_path.stem.startswith("robot")
    assert (mock_cache_dir / "repos").exists()


def test_package_path_replacement(temp_git_repo, mock_cache_dir):
    """Test that package:// paths are replaced with relative paths."""
    repo_path = temp_git_repo
    urdf_content = """<robot name="test">
    <link name="base">
        <visual>
            <geometry>
                <mesh filename="package://test_repo/meshes/base.stl"/>
            </geometry>
        </visual>
    </link>
</robot>"""
    (repo_path / "robot_pkg.urdf").write_text(urdf_content)
    repo = git.Repo(repo_path)
    repo.index.add(["robot_pkg.urdf"])
    repo.index.commit("Add pkg urdf")

    repo_url = f"file://{repo_path}"
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot_pkg.urdf", cache_dir=mock_cache_dir
    )

    content = asset_path.read_text()
    assert "package://test_repo/" not in content
    # Should be replaced by path relative to repo root
    assert "meshes/base.stl" in content


def test_xacro_conversion(temp_git_repo, mock_cache_dir):
    """Test that xacro files are automatically converted to URDF."""
    repo_url = f"file://{temp_git_repo}"
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot.xacro", cache_dir=mock_cache_dir
    )
    assert asset_path.suffix == ".urdf"
    assert "test_xacro" in asset_path.read_text()


def test_process_xacro(temp_git_repo):
    """Test process_xacro directly."""
    xacro_path = temp_git_repo / "robot.xacro"
    urdf_xml = ram.process_xacro(xacro_path)
    assert "test_xacro" in urdf_xml
    assert "package://" not in urdf_xml
