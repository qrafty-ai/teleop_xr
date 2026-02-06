
## Contract Test Findings (Feb 06 2026)
- Pyroki already has `RobotCollision.from_sphere_decomposition` implemented (version 0.0.0).
- The schema expected by Pyroki matches the one in our contract: `{"link_name": {"centers": [[x,y,z], ...], "radii": [r, ...]}}`.
- Robot initialization tests confirm that `TeaArmRobot` (and likely others) currently use `from_urdf` by default and do not yet process `sphere_decomposition` parameters.
- Created `tests/test_collision_sphere_contract.py` with RED tests for:
    - `teleop_xr.ik.collision.validate_sphere_decomposition`
    - `teleop_xr.ik.collision.get_decomposition_cache_key`
### Deterministic Decomposition Cache
- Implemented  in .
- Uses  for process-safe access to JSON cache files.
- Cache key is a SHA256 hash of URDF hash, mesh fingerprints, package versions (ballpark, pyroki), and parameters.
- Metadata is stored alongside data to track versioning and allow for explicit invalidation.
- User-facing messages are logged using  to provide feedback on cache hits and invalidations.

### Deterministic Decomposition Cache
- Implemented CollisionSphereCache in teleop_xr/ik/collision_sphere_cache.py.
- Uses filelock for process-safe access to JSON cache files.
- Cache key is a SHA256 hash of URDF hash, mesh fingerprints, package versions (ballpark, pyroki), and parameters.
- Metadata is stored alongside data to track versioning and allow for explicit invalidation.
- User-facing messages are logged using loguru to provide feedback on cache hits and invalidations.

## Sphere Collision Decomposition Learnings

### Implementation Patterns
- **Analytical Fallback**: For simple primitives like single spheres, analytical calculation is much more accurate and efficient than mesh-based decomposition.
- **Strict Over-approximation**: Ballpark is a sampling-based algorithm. To ensure strict coverage of ALL mesh vertices, a combination of ballpark's internal padding (multiplier) and a post-check with a small extra margin (e.g., 5%) and epsilon (e.g., 1e-3) is necessary.
- **URDF Mesh Loading**: `yourdfpy` resolves paths via `_filename_handler`, but meshes must be loaded manually via `trimesh.load` as they are not automatically attached to the `Mesh` geometry objects in a way that's easily accessible without building the whole scene.
- **Deterministic Results**: Ballpark is stochastic. Seeding both `numpy` and `jax` (if possible) is important for reproducible results. Sorting the final sphere list by coordinates ensures consistency in the output schema.

### Tooling Gotchas
- **Trimesh Bounding Sphere**: `mesh.bounding_sphere` returns a `trimesh.primitives.Sphere` object, which stores its properties in the `.primitive` attribute (e.g., `.primitive.radius`).
- **JAX Arrays**: Ballpark returns JAX arrays which should be converted to standard numpy arrays or Python floats/lists for JSON serialization and general compatibility.

## Integration of Sphere Decomposition in Robots
- Integrated `CollisionSphereCache` and `generate_collision_spheres` into `h1_2` and `teaarm` robot classes.
- Used `RobotCollision.from_sphere_decomposition` with a fallback to `from_urdf` if the decomposition is empty. This handles robots with no collision geometry (common in tests) gracefully, avoiding `AssertionError` from pyroki.
- Re-enabled self-collision cost terms in `build_costs` for both robots.
- Updated `tests/test_robots.py` to expect an additional cost term for `teaarm` since self-collision is now always active.
- Verified that cache hits/misses are correctly logged using loguru.

## Task 5: GREEN coverage for verification hardening
- Implemented strict coverage tests using `trimesh` to verify that all mesh vertices are contained within the generated spheres.
- Verified fallback mechanisms: if `ballpark` fails or the resulting spheres don't cover the mesh, the system correctly falls back to a single bounding sphere.
- Tested log message sequences to ensure user-visible feedback matches expectations for cache hits and misses/invalidations.
- Confirmed deterministic behavior of cache keys and sphere ordering within links.
- Verified integration with H1 and TeaArm robots.

## Parameter Pinning & Refactoring (Task 6)
- Centralized all decomposition parameters into `CollisionConfig` dataclass in `teleop_xr/ik/collision.py`.
- This ensures that cache keys are deterministic and consistent across different robot implementations.
- Ballpark parameters (`percentile`, `padding` multiplier, `n_samples`) are now explicit.
- Added legacy support via `**kwargs` in `generate_collision_spheres` to maintain test compatibility while encouraging the new config-based API.
- Verified that Franka implementation remains untouched as it uses default mesh-based collision.
