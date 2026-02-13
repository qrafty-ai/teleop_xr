"""
Robot Asset Manager (RAM) module.
Handles fetching and processing robot descriptions (URDF/Xacro) from git.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

import git
import xacro
import xacro.substitution_args
from filelock import FileLock
import trimesh


# --- Xacro Patching for RAM ---
_CURRENT_REPO_ROOT: Optional[Path] = None


def _resolve_package(package_name: str) -> str:
    """Resolve a package name to a path within the current RAM repo."""
    if _CURRENT_REPO_ROOT:
        # 1. Check root
        candidate = _CURRENT_REPO_ROOT / package_name
        if candidate.exists():
            return candidate.as_posix()

        # 2. Check immediate subdirectories (common for metapackages)
        for child in _CURRENT_REPO_ROOT.iterdir():
            if child.is_dir():
                if child.name == package_name:  # pragma: no cover
                    return child.as_posix()
                # Check one level deeper
                candidate = child / package_name
                if candidate.exists():
                    return candidate.as_posix()

        # 3. Check if repo root itself IS the package
        #    (common for single-package repos like openarm_description)
        #    Also check if repo root starts with package_name (handles hash suffixes)
        if (_CURRENT_REPO_ROOT.name == package_name or
            _CURRENT_REPO_ROOT.name.startswith(package_name + "_")):
            return _CURRENT_REPO_ROOT.as_posix()

        # 4. Check if repo root contains a package.xml with the matching name
        #    This handles cases where the repo folder has a hash suffix (e.g. repo_hash)
        #    but contains the package we are looking for.
        package_xml = _CURRENT_REPO_ROOT / "package.xml"
        if package_xml.exists():
            try:
                content = package_xml.read_text(encoding='utf-8')
                match = re.search(r"<name>\s*([^<\s]+)\s*</name>", content)
                if match and match.group(1) == package_name:
                    return _CURRENT_REPO_ROOT.as_posix()
            except Exception:
                pass  # Ignore parsing errors

    # Fallback: ignore or raise?
    raise ValueError(
        f"Package '{package_name}' not found in RAM repo {_CURRENT_REPO_ROOT}"
    )


def _mock_eval_find(package_name):
    """Mock implementation of $(find pkg) for xacro."""
    # xacro calls _eval_find(pkg) directly, passing the string.
    return _resolve_package(package_name)


# Apply patch
xacro.substitution_args._eval_find = _mock_eval_find


@contextmanager
def _ram_repo_context(repo_root: Path):
    """Context manager to set the current repo root for xacro resolution."""
    global _CURRENT_REPO_ROOT
    old_root = _CURRENT_REPO_ROOT
    _CURRENT_REPO_ROOT = repo_root
    try:
        yield
    finally:
        _CURRENT_REPO_ROOT = old_root


# --- End Patching ---


def get_cache_root() -> Path:
    """Get the root directory for RAM cache."""
    cache_root = Path.home() / ".cache" / "ram"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _get_repo_dir(repo_url: str, cache_dir: Path) -> Path:
    """Get the directory name for a repo based on its URL hash."""
    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    # Try to extract a human readable name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    return cache_dir / "repos" / f"{repo_name}_{url_hash}"


def _replace_package_uris(urdf_content: str, repo_root: Path) -> str:
    """
    Replace package:// URI with absolute paths to files in the repository.
    """

    def resolve_uri(match):
        # match.group(1) is the package name
        sub_path = match.group(2)
        # Try to resolve package path first
        try:
            pkg_path = Path(_resolve_package(match.group(1)))
            # Use as_posix() to ensure forward slashes, important for Windows
            return (pkg_path / sub_path).resolve().as_posix()
        except ValueError:
            # Fallback to simple root join if resolution fails
            return (repo_root / sub_path).resolve().as_posix()

    return re.sub(r"package://([^/]+)/(.*)", resolve_uri, urdf_content)


def _convert_dae_to_glb(dae_path: Path) -> Path:
    """Convert a .dae file to .glb and return the new path."""
    glb_path = dae_path.with_suffix(".glb")
    if glb_path.exists():
        return glb_path

    try:
        scene = trimesh.load(dae_path, force="scene")
        # Ensure we export as GLB
        glb_data = scene.export(file_obj=None, file_type="glb")  # type: ignore
        if isinstance(glb_data, bytes):
            glb_path.write_bytes(glb_data)
        else:
            with open(glb_path, "wb") as f:
                f.write(glb_data)  # type: ignore
        return glb_path
    except Exception as e:
        print(f"Failed to convert {dae_path} to GLB: {e}")
        return dae_path


def _replace_dae_with_glb(urdf_content: str) -> str:
    """Find all .dae file references in the URDF, convert them to .glb, and update the URDF content to point to the .glb files."""

    def replace_match(match: re.Match[str]) -> str:
        original_path_str = match.group(1)
        original_path = Path(original_path_str)

        if original_path.suffix.lower() == ".dae" and original_path.exists():
            glb_path = _convert_dae_to_glb(original_path)
            if glb_path != original_path:
                return f'filename="{glb_path.as_posix()}"'

        return f'filename="{Path(original_path_str).as_posix()}"'

    return re.sub(r'filename="([^"]+)"', replace_match, urdf_content)


def get_repo(
    repo_url: str, branch: Optional[str] = None, cache_dir: Optional[Path] = None
) -> Path:
    """
    Fetch a git repository into the cache and return its local path.
    """
    if cache_dir is None:
        cache_dir = get_cache_root()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = _get_repo_dir(repo_url, cache_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    lock_path = repo_dir.with_suffix(".lock")

    with FileLock(lock_path):
        if not repo_dir.exists():
            # Clone repo
            repo = git.Repo.clone_from(repo_url, repo_dir, branch=branch)
            repo.close()
        else:
            # Update repo
            repo = git.Repo(repo_dir)
            try:
                if branch:
                    try:
                        repo.git.checkout(branch)
                    except git.GitCommandError:
                        # If checkout fails, try fetching first
                        repo.remotes.origin.fetch()
                        repo.git.checkout(branch)

                repo.remotes.origin.pull()
            finally:
                repo.close()

    return repo_dir


def process_xacro(
    xacro_path: Path,
    repo_root: Path,
    mappings: Optional[Dict[str, str]] = None,
    resolve_packages: bool = True,
) -> str:
    """
    Process a xacro file and return the URDF XML string.
    If resolve_packages is True, package:// URIs are resolved to absolute paths.
    """
    # Use context to allow $(find ...) resolution
    with _ram_repo_context(repo_root):
        doc: Any = xacro.process_file(str(xacro_path), mappings=mappings)
        urdf_xml = doc.toprettyxml(indent="  ")

    if resolve_packages:
        # Also use context here for better package:// resolution
        with _ram_repo_context(repo_root):
            return _replace_package_uris(urdf_xml, repo_root)
    return urdf_xml


def get_resource(
    repo_url: Optional[str] = None,
    path_inside_repo: str = "",
    repo_root: Optional[Path] = None,
    branch: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    xacro_args: Optional[Dict[str, str]] = None,
    resolve_packages: bool = True,
    convert_dae_to_glb: bool = False,
) -> Path:
    """
    Main entry point for fetching a robot resource.
    """
    if not repo_url and not repo_root:
        raise ValueError("Either repo_url or repo_root must be provided")
    if repo_url and repo_root:
        raise ValueError("Only one of repo_url or repo_root can be provided")

    if Path(path_inside_repo).is_absolute():
        raise ValueError("path_inside_repo must be relative")

    if cache_dir is None:
        cache_dir = get_cache_root()

    if repo_root:
        repo_dir = Path(repo_root)
    else:
        assert repo_url is not None
        repo_dir = get_repo(repo_url, branch=branch, cache_dir=cache_dir)

    file_path = repo_dir / path_inside_repo

    if not file_path.exists():
        msg = f"Asset {path_inside_repo} not found in "
        msg += f"local path {repo_root}" if repo_root else f"repo {repo_url}"
        raise FileNotFoundError(msg)

    is_xacro = file_path.suffix == ".xacro" or ".urdf.xacro" in file_path.name

    if is_xacro:
        # Unique name for output based on args and resolution
        arg_str = str(sorted(xacro_args.items())) if xacro_args else ""
        arg_str += f"_resolved={resolve_packages}"
        arg_str += f"_glb={convert_dae_to_glb}"

        if repo_root:
            # Hash includes absolute repo path to avoid collisions
            arg_str += f"_root={repo_dir.absolute()}"
            arg_str += f"_path={path_inside_repo}"

        arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()[:12]

        if repo_root:
            output_dir = cache_dir / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_urdf_path = output_dir / f"{file_path.stem}_{arg_hash}.urdf"
        else:
            output_urdf_path = (
                file_path.parent / f"{file_path.stem}_{arg_hash[:6]}.urdf"
            )

        # Process xacro
        urdf_content = process_xacro(
            file_path, repo_dir, mappings=xacro_args, resolve_packages=resolve_packages
        )

        if convert_dae_to_glb:
            urdf_content = _replace_dae_with_glb(urdf_content)

        output_urdf_path.write_text(urdf_content)
    else:
        # For plain URDF
        if resolve_packages or convert_dae_to_glb:
            content = file_path.read_text()
            needs_processing = False

            if resolve_packages and "package://" in content:
                needs_processing = True
            if convert_dae_to_glb and ".dae" in content:
                needs_processing = True

            if needs_processing:
                if repo_root:
                    # Create a processed version in cache for local repos too
                    output_dir = cache_dir / "processed"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Use a hash to avoid collisions
                    path_hash = hashlib.sha256(
                        f"{repo_dir.absolute()}:{path_inside_repo}:{convert_dae_to_glb}".encode()
                    ).hexdigest()[:12]
                    output_urdf_path = output_dir / f"{file_path.stem}_{path_hash}.urdf"
                else:
                    suffix = "_processed_glb" if convert_dae_to_glb else "_processed"
                    output_urdf_path = (
                        file_path.parent / f"{file_path.stem}{suffix}.urdf"
                    )

                with _ram_repo_context(repo_dir):
                    if resolve_packages:
                        content = _replace_package_uris(content, repo_dir)

                    if convert_dae_to_glb:
                        content = _replace_dae_with_glb(content)

                output_urdf_path.write_text(content)
            else:
                output_urdf_path = file_path
        else:
            output_urdf_path = file_path

    return output_urdf_path


# Alias get_asset to get_resource for backward compatibility if any
get_asset = get_resource
