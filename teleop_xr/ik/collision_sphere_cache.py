import hashlib
import json
from typing import Dict, Any, Optional
from filelock import FileLock
from loguru import logger
from teleop_xr.ram import get_cache_root


class CollisionSphereCache:
    """
    Deterministic decomposition cache for collision spheres with lock-safe persistence.
    """

    def __init__(self, robot_slug: str):
        self.robot_slug = robot_slug
        self.cache_dir = get_cache_root() / "collision_spheres" / robot_slug
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.cache_dir / "cache.lock"

    def _get_version(self, package_name: str) -> str:
        """Get the version of a package."""
        try:
            import importlib.metadata

            return importlib.metadata.version(package_name)
        except Exception:
            try:
                module = __import__(package_name)
                return getattr(module, "__version__", "unknown")
            except Exception:
                return "unknown"

    def compute_cache_key(
        self, urdf_hash: str, mesh_fingerprints: Dict[str, str], params: Dict[str, Any]
    ) -> str:
        """
        Compute a deterministic cache key based on URDF, meshes, versions, and parameters.
        """
        ballpark_version = self._get_version("ballpark")
        pyroki_version = self._get_version("pyroki")

        # Ensure dicts are sorted for deterministic JSON serialization
        mesh_str = json.dumps(mesh_fingerprints, sort_keys=True)
        params_str = json.dumps(params, sort_keys=True)

        combined = (
            f"urdf:{urdf_hash}|"
            f"meshes:{mesh_str}|"
            f"ballpark:{ballpark_version}|"
            f"pyroki:{pyroki_version}|"
            f"params:{params_str}"
        )
        return hashlib.sha256(combined.encode()).hexdigest()

    def load(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load cached collision spheres if they exist and are valid.
        """
        data_path = self.cache_dir / f"{cache_key}.json"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        with FileLock(self.lock_path):
            if not data_path.exists() or not meta_path.exists():
                # We don't log "invalidated" here because it might just be missing (first time)
                return None

            try:
                with open(data_path, "r") as f:
                    data = json.load(f)
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                # Verify metadata consistency
                if meta.get("cache_key") != cache_key:
                    logger.warning(
                        f"Collision sphere cache invalidated for {self.robot_slug}: corrupt (key mismatch)."
                    )
                    return None

                # Check versions in meta against current versions
                ballpark_version = self._get_version("ballpark")
                pyroki_version = self._get_version("pyroki")

                if (
                    meta.get("ballpark_version") != ballpark_version
                    or meta.get("pyroki_version") != pyroki_version
                ):
                    logger.warning(
                        f"Collision sphere cache invalidated for {self.robot_slug}: stale (version mismatch)."
                    )
                    return None

                logger.info(f"Loaded cached collision spheres for {self.robot_slug}.")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"Collision sphere cache invalidated for {self.robot_slug}: corrupt ({e})."
                )
                return None

    def save(self, cache_key: str, data: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Save collision spheres and metadata to cache.
        """
        data_path = self.cache_dir / f"{cache_key}.json"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"

        # Enrich metadata
        metadata["cache_key"] = cache_key
        metadata["ballpark_version"] = self._get_version("ballpark")
        metadata["pyroki_version"] = self._get_version("pyroki")
        metadata["robot_slug"] = self.robot_slug

        with FileLock(self.lock_path):
            try:
                # Save data
                with open(data_path, "w") as f:
                    json.dump(data, f, indent=2)
                # Save metadata
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            except IOError as e:
                logger.error(
                    f"Failed to save collision sphere cache for {self.robot_slug}: {e}"
                )

    def invalidate(self, reason: str):
        """
        Explicitly invalidate the cache for this robot.
        """
        logger.warning(
            f"Collision sphere cache invalidated for {self.robot_slug}: {reason}."
        )
        with FileLock(self.lock_path):
            for p in self.cache_dir.glob("*.json"):
                try:
                    p.unlink()
                except OSError:
                    pass
