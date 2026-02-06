import pytest
import json
from teleop_xr.ik.collision_sphere_cache import CollisionSphereCache
from teleop_xr.ik.collision import get_decomposition_cache_key
from unittest.mock import patch
from loguru import logger


@pytest.fixture
def temp_cache_dir(tmp_path):
    # Mock get_cache_root to use the temporary path
    with patch(
        "teleop_xr.ik.collision_sphere_cache.get_cache_root", return_value=tmp_path
    ):
        yield tmp_path


def test_cache_key_determinism(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    urdf_hash = "hash1"
    mesh_fingerprints = {"link1": "mesh_hash1", "link2": "mesh_hash2"}
    params = {"resolution": 0.01, "padding": 0.005}

    # Mock versions to ensure stability across environments
    with patch.object(
        cache,
        "_get_version",
        side_effect=lambda pkg: "1.2.3" if pkg == "ballpark" else "0.1.0",
    ):
        key1 = cache.compute_cache_key(urdf_hash, mesh_fingerprints, params)

        # Different order of dicts should yield same key
        mesh_fingerprints_alt = {"link2": "mesh_hash2", "link1": "mesh_hash1"}
        params_alt = {"padding": 0.005, "resolution": 0.01}
        key2 = cache.compute_cache_key(urdf_hash, mesh_fingerprints_alt, params_alt)

        assert key1 == key2

        # Different values should yield different key
        key3 = cache.compute_cache_key(
            urdf_hash, mesh_fingerprints, {"resolution": 0.02}
        )
        assert key1 != key3


def test_save_load(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    key = "test_key"
    data = {"spheres": [{"center": [0, 0, 0], "radius": 0.1}]}
    meta = {"source": "test"}

    cache.save(key, data, meta)

    loaded_data = cache.load(key)
    assert loaded_data == data

    # Check if files exist
    assert (
        temp_cache_dir / "collision_spheres" / "test_robot" / f"{key}.json"
    ).exists()
    assert (
        temp_cache_dir / "collision_spheres" / "test_robot" / f"{key}.meta.json"
    ).exists()


def test_stale_version_invalidation(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    key = "test_key"
    data = {"spheres": []}
    meta = {"source": "test"}

    # Mock versions to "1.0.0"
    with patch.object(cache, "_get_version", return_value="1.0.0"):
        cache.save(key, data, meta)

    # Try to load with version "2.0.0"
    with patch.object(cache, "_get_version", return_value="2.0.0"):
        loaded_data = cache.load(key)
        assert loaded_data is None


def test_corrupt_file_invalidation(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    key = "test_key"
    data = {"spheres": []}
    meta = {"source": "test"}

    cache.save(key, data, meta)

    # Corrupt the data file
    data_path = temp_cache_dir / "collision_spheres" / "test_robot" / f"{key}.json"
    data_path.write_text("invalid json")

    loaded_data = cache.load(key)
    assert loaded_data is None


def test_key_mismatch_invalidation(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    key = "test_key"
    data = {"spheres": []}
    meta = {"source": "test"}

    cache.save(key, data, meta)

    # Modify meta to have wrong key
    meta_path = temp_cache_dir / "collision_spheres" / "test_robot" / f"{key}.meta.json"
    with open(meta_path, "r") as f:
        meta_content = json.load(f)
    meta_content["cache_key"] = "wrong_key"
    with open(meta_path, "w") as f:
        json.dump(meta_content, f)

    loaded_data = cache.load(key)
    assert loaded_data is None


def test_explicit_invalidate(temp_cache_dir):
    cache = CollisionSphereCache("test_robot")
    cache.save("key1", {"d": 1}, {})
    cache.save("key2", {"d": 2}, {})

    robot_dir = temp_cache_dir / "collision_spheres" / "test_robot"
    assert len(list(robot_dir.glob("*.json"))) == 4  # 2 data + 2 meta

    cache.invalidate("testing")
    assert len(list(robot_dir.glob("*.json"))) == 0


def test_decomposition_key_determinism():
    # Test that get_decomposition_cache_key is deterministic despite float precision
    decomp1 = {"link1": {"centers": [[0.1000000001, 0, 0]], "radii": [0.1000000001]}}
    decomp2 = {"link1": {"centers": [[0.1, 0, 0]], "radii": [0.1]}}
    key1 = get_decomposition_cache_key(decomp1)
    key2 = get_decomposition_cache_key(decomp2)
    assert key1 == key2


def test_log_messages(temp_cache_dir):
    messages = []

    def sink(msg):
        messages.append(msg.record["message"])

    handler_id = logger.add(sink, format="{message}")

    try:
        cache = CollisionSphereCache("test_robot")
        key = "test_key"
        data = {"spheres": []}
        meta = {
            "source": "test",
            "ballpark_version": "1.0.0",
            "pyroki_version": "0.1.0",
        }

        # Mock versions to "1.0.0" and "0.1.0"
        with patch.object(
            cache,
            "_get_version",
            side_effect=lambda pkg: "1.0.0" if pkg == "ballpark" else "0.1.0",
        ):
            cache.save(key, data, meta)

            # Verify log message on successful load
            loaded = cache.load(key)
            assert loaded is not None
            assert any(
                "Loaded cached collision spheres for test_robot" in m for m in messages
            )

            # Verify log message on stale version invalidation
            messages.clear()
            with patch.object(cache, "_get_version", return_value="2.0.0"):
                loaded = cache.load(key)
                assert loaded is None
                assert any("Collision sphere cache invalidated" in m for m in messages)
                assert any("stale" in m for m in messages)
    finally:
        logger.remove(handler_id)
