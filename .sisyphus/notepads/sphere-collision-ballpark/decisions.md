# Decisions - Sphere Collision Ballpark

## Decision: Centralized Collision Configuration

- **Rationale**: Previously, robots hardcoded their own decomposition parameters
  (`n_spheres_per_link=8`, etc.). To ensure reliability and reproducibility,
  these are now managed by `CollisionConfig`.
- **Implementation**: Used a frozen dataclass for `CollisionConfig` to ensure it
  can be easily hashed or converted to a stable dict for cache key generation.
- **Compatibility**: Maintained `**kwargs` support in
  `generate_collision_spheres` to avoid breaking existing tests while
  transitioning robots to the new pattern.
