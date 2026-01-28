from pathlib import Path

from teleop import _resolve_frontend_paths


def test_resolve_frontend_paths_prefers_dist(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "teleop"
    dist_dir = repo_root / "webxr" / "dist"

    package_dir.mkdir(parents=True)
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("dist")

    static_dir, index_path, mount_path, mount_name = _resolve_frontend_paths(str(package_dir))

    assert Path(static_dir) == dist_dir
    assert Path(index_path) == dist_dir / "index.html"
    assert mount_path == "/"
    assert mount_name == "webxr"


def test_resolve_frontend_paths_fallback(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    package_dir = repo_root / "teleop"
    assets_dir = package_dir / "assets"

    assets_dir.mkdir(parents=True)
    (package_dir / "index.html").write_text("fallback")

    static_dir, index_path, mount_path, mount_name = _resolve_frontend_paths(str(package_dir))

    assert Path(static_dir) == assets_dir
    assert Path(index_path) == package_dir / "index.html"
    assert mount_path == "/assets"
    assert mount_name == "assets"
