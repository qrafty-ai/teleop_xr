import pytest
from teleop_xr import ram


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Fixture to provide a temporary cache directory for RAM."""
    cache_dir = tmp_path / "ram_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def local_repo(tmp_path):
    """Fixture to create a local directory with dummy assets (not a git repo)."""
    repo_path = tmp_path / "local_assets"
    repo_path.mkdir()

    # Create dummy assets
    urdf_path = repo_path / "robot.urdf"
    urdf_path.write_text('<robot name="test_robot"><link name="base_link"/></robot>')

    xacro_path = repo_path / "robot.xacro"
    xacro_path.write_text(
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="test_xacro"><link name="base"/></robot>'
    )

    mesh_dir = repo_path / "meshes"
    mesh_dir.mkdir()
    (mesh_dir / "base.stl").write_text("dummy mesh content")

    return repo_path


def test_get_resource_local_urdf(local_repo, mock_cache_dir):
    """Test loading a local URDF resource."""
    asset_path = ram.get_resource(
        path_inside_repo="robot.urdf", repo_root=local_repo, cache_dir=mock_cache_dir
    )
    assert asset_path.exists()
    assert asset_path.suffix == ".urdf"
    assert "test_robot" in asset_path.read_text()
    # For plain URDF with no package://, it might return the original path
    # But if we resolve packages, it might create a processed one.
    # In this case, no package:// so it should be the original path if resolve_packages doesn't find any.


def test_get_resource_local_xacro(local_repo, mock_cache_dir):
    """Test loading a local Xacro resource."""
    asset_path = ram.get_resource(
        path_inside_repo="robot.xacro", repo_root=local_repo, cache_dir=mock_cache_dir
    )
    assert asset_path.exists()
    assert asset_path.suffix == ".urdf"
    assert "test_xacro" in asset_path.read_text()

    # Check that it's in the processed cache, not the local repo
    assert str(mock_cache_dir / "processed") in str(asset_path)
    assert str(local_repo) not in str(asset_path)


def test_get_resource_local_xacro_args_caching(local_repo, mock_cache_dir):
    """Test that different xacro args result in different cached files."""
    path1 = ram.get_resource(
        path_inside_repo="robot.xacro",
        repo_root=local_repo,
        cache_dir=mock_cache_dir,
        xacro_args={"arg1": "val1"},
    )
    path2 = ram.get_resource(
        path_inside_repo="robot.xacro",
        repo_root=local_repo,
        cache_dir=mock_cache_dir,
        xacro_args={"arg1": "val2"},
    )
    assert path1 != path2
    assert path1.name != path2.name


def test_get_resource_absolute_path_error(local_repo, mock_cache_dir):
    """Test that path_inside_repo must be relative."""
    with pytest.raises(ValueError, match="path_inside_repo must be relative"):
        ram.get_resource(
            path_inside_repo="/absolute/path.urdf",
            repo_root=local_repo,
            cache_dir=mock_cache_dir,
        )


def test_get_resource_missing_both_error(mock_cache_dir):
    """Test error when both repo_url and repo_root are missing."""
    with pytest.raises(
        ValueError, match="Either repo_url or repo_root must be provided"
    ):
        ram.get_resource(path_inside_repo="robot.urdf", cache_dir=mock_cache_dir)


def test_get_resource_both_provided_error(local_repo, mock_cache_dir):
    """Test error when both repo_url and repo_root are provided."""
    with pytest.raises(
        ValueError, match="Only one of repo_url or repo_root can be provided"
    ):
        ram.get_resource(
            repo_url="https://github.com/user/repo.git",
            path_inside_repo="robot.urdf",
            repo_root=local_repo,
            cache_dir=mock_cache_dir,
        )
