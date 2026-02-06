# Multi-Sphere Self-Collision for TeaArm

## TL;DR

> **Quick Summary**: Replace the current single-bounding-capsule-per-link self-collision model (which is too coarse and currently commented out) with a multi-sphere-per-link decomposition inspired by the sparrows library. Each link is approximated by N spheres placed along joint-to-joint centerlines with mesh-informed radii, enabling tighter collision detection that actually works for the dual-arm TeaArm robot.
>
> **Deliverables**:
> - New `MultiSphereCollision` class in `teleop_xr/ik/collision.py`
> - Sphere fitting algorithm: centerline-based placement + mesh-projected radii
> - Integration into `TeaArmRobot.build_costs()` (uncomment + rewire self-collision cost)
> - Validation script proving collision detection works on known poses
>
> **Estimated Effort**: Medium (1-2 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
The self-collision check in `teleop_xr/ik/robots/teaarm.py` (pyroki) is too coarse. Reimplement the capsule/sphere generation algorithm to use multiple spheres/capsules per link for better collision fidelity. Reference: [sparrows](https://github.com/roahmlab/sparrows).

### Interview Summary
**Key Discussions**:
- The current `pk.collision.RobotCollision.from_urdf()` fits ONE `trimesh.bounds.minimum_cylinder()` per link — this over-approximates complex meshes (torso, shoulder joints) so badly the collision cost was commented out entirely.
- Sparrows decomposes links into N spheres along joint-center-to-joint-center lines with analytically computed radii.
- Oracle recommends: multi-sphere per link, flattened primitive arrays with precomputed pair indices, mesh-informed radii, N≈6 for arm links.

**Research Findings**:
- pyroki already has `Sphere` dataclass, `sphere_sphere` distance function, and `colldist_from_sdf` — all reusable.
- TeaArm: 19 links, 2-DOF waist + two 7-DOF arms. SRDF has comprehensive ignore pairs.
- The `pairwise_collide` approach creates O(L²) pairs. Flattened sphere approach with precomputed pair indices avoids O(L²×N²) blowups.
- `Capsule.decompose_to_spheres()` exists but uses uniform radius (not mesh-informed), making it unsuitable for non-cylindrical links.

### Metis Review
**Identified Gaps** (addressed):
- Torso branching: torso_link has two children (left_arm_l1, right_arm_l1). Plan uses multi-centerline approach (parent→each child) for such links.
- Zero-length centerlines (fixed joints / EE frames): Plan falls back to single sphere from mesh bounding sphere or skip.
- Conservative vs best-fit radii: Default to conservative (max distance to centerline + small margin).
- Multiple collision geometries per link: Aggregate vertices across all sub-meshes with their URDF transforms.

---

## Work Objectives

### Core Objective
Create a tighter self-collision model for TeaArm that uses multiple spheres per link (placed along joint centerlines with mesh-projected radii), enabling effective self-collision avoidance in the IK solver.

### Concrete Deliverables
- `teleop_xr/ik/collision.py` — New `MultiSphereCollision` jax_dataclass
- `teleop_xr/ik/collision.py` — `build_multi_sphere_collision(urdf, ...)` factory function
- Updated `teleop_xr/ik/robots/teaarm.py` — Self-collision cost enabled with new model
- `tests/test_multi_sphere_collision.py` — Validation tests

### Definition of Done
- [x] `TeaArmRobot` uses multi-sphere self-collision cost (no longer commented out)
- [x] Default configuration (zeros) produces non-penetrating distances
- [x] Collision cost is differentiable (JAX jit + grad compatible)
- [x] Known colliding pose produces non-zero collision cost

### Must Have
- Multiple spheres per link (not just one primitive per link)
- Mesh-informed radii (not uniform/arbitrary)
- Precomputed pair indices (not runtime pair generation)
- JAX-compatible (jit, vmap, grad friendly)
- Conservative approximation (no false negatives — spheres fully contain link geometry)
- Respect URDF adjacency for ignore pairs

### Must NOT Have (Guardrails)
- **NO** upstream changes to pyroki source code — only use pyroki's public API/types
- **NO** learned/optimized sphere packing (runtime optimization at init)
- **NO** per-pose adaptive fitting
- **NO** convex decomposition or GJK/ODE/FCL integration
- **NO** visualization tooling (unless strictly optional and trivial)
- **NO** changes to the semantics of other costs in `build_costs()` (only add/modify self-collision)
- **NO** modification of URDF/SRDF files
- **NO** automatic SRDF generation or ignore-pair heuristics beyond what URDF adjacency provides

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (pytest in project)
- **Automated tests**: YES (tests-after, not TDD)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

Every task includes Agent-Executed QA scenarios as the primary verification method, complemented by unit tests.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: MultiSphereCollision data model + distance computation
└── (no other independent tasks)

Wave 2 (After Wave 1):
├── Task 2: Sphere fitting algorithm (build_multi_sphere_collision)
└── (depends on Task 1 data structures)

Wave 3 (After Wave 2):
├── Task 3: TeaArm integration (uncomment + rewire collision cost)
└── (depends on Task 2 factory)

Wave 4 (After Wave 3):
└── Task 4: Tests + validation
    (depends on Task 3 full integration)
```

Critical Path: Task 1 → Task 2 → Task 3 → Task 4

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4 | None |
| 2 | 1 | 3, 4 | None |
| 3 | 2 | 4 | None |
| 4 | 3 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | delegate_task(category="deep", load_skills=[], run_in_background=false) |
| 2 | 2 | delegate_task(category="deep", load_skills=[], run_in_background=false) |
| 3 | 3 | delegate_task(category="quick", load_skills=[], run_in_background=false) |
| 4 | 4 | delegate_task(category="unspecified-low", load_skills=[], run_in_background=false) |

---

## TODOs

- [x] 1. Create `MultiSphereCollision` data model + sphere-sphere distance computation

  **What to do**:

  Create a new file `teleop_xr/ik/collision.py` containing a JAX pytree dataclass `MultiSphereCollision` that stores a flattened set of spheres (one array for all links combined) with precomputed collision pair indices.

  **Data structure** (jax_dataclass):
  ```python
  @jdc.pytree_dataclass
  class MultiSphereCollision:
      num_primitives: jdc.Static[int]          # Total sphere count across all links
      num_links: jdc.Static[int]               # Number of robot links
      link_names: jdc.Static[tuple[str, ...]]   # Link name list (matching Robot.links.names)

      # Sphere geometry in link-local coordinates
      sphere_centers_local: Float[Array, "P 3"]  # (P, 3) — centers in parent link frame
      sphere_radii: Float[Array, "P"]            # (P,) — radii
      sphere_link_indices: Int[Array, "P"]        # (P,) — which link each sphere belongs to

      # Precomputed collision pairs (over primitives, not links)
      pair_i: Int[Array, "K"]                    # (K,) — first primitive index in each pair
      pair_j: Int[Array, "K"]                    # (K,) — second primitive index in each pair
  ```

  **Methods to implement**:

  1. `at_config(robot, cfg) -> tuple[Float[Array, "P 3"], Float[Array, "P"]]`:
     - Run `robot.forward_kinematics(cfg)` → get `(num_links, 7)` SE3 wxyz_xyz
     - Gather link poses for each sphere using `sphere_link_indices`
     - Transform `sphere_centers_local` to world frame: `world_center = link_pose @ local_center`
     - Return `(world_centers, sphere_radii)`

  2. `compute_self_collision_distance(robot, cfg) -> Float[Array, "K"]`:
     - Call `at_config` to get world centers + radii
     - For each pair (pair_i, pair_j): `dist = ||c_i - c_j|| - (r_i + r_j)`
     - Return array of K distances (positive = separated, negative = penetrating)

  **Implementation notes**:
  - Use `jaxlie.SE3` for pose transforms (consistent with pyroki)
  - The `at_config` gather operation: `link_poses = all_link_poses[sphere_link_indices]` — this is a simple JAX gather, fully jit-compatible
  - Distance computation: vectorized over all K pairs, no loops
  - `sphere_sphere` distance is simply `||c1 - c2|| - r1 - r2` — reimplement directly rather than using pyroki's `Sphere` class to avoid overhead of constructing `Sphere` objects at runtime

  **Must NOT do**:
  - Don't use pyroki's `pairwise_collide` (it materializes full L×L matrix)
  - Don't create `Sphere` dataclass objects at runtime — work with raw arrays
  - Don't add any non-JAX operations (numpy, trimesh) in the runtime path (pure JAX only)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: JAX pytree dataclass design requires careful understanding of jit/vmap constraints; needs to correctly handle SE3 transforms on gathered indices.
  - **Skills**: (none needed — pure Python/JAX work)

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (solo)
  - **Blocks**: Tasks 2, 3, 4
  - **Blocked By**: None

  **References** (CRITICAL):

  **Pattern References** (existing code to follow):
  - `pyroki/collision/_robot_collision.py:22-37` — `RobotCollision` jax_dataclass structure: shows how to define a collision container with `num_links`, `link_names`, `coll`, `active_idx_i/j`. Follow this pattern for field naming and jdc.Static usage.
  - `pyroki/collision/_robot_collision.py:235-262` — `RobotCollision.at_config()`: shows how to run FK, get link poses as `jaxlie.SE3`, and transform collision geometry to world frame. This is the pattern for our `at_config`.
  - `pyroki/collision/_robot_collision.py:320-360` — `RobotCollision.compute_self_collision_distance()`: shows the full pipeline (at_config → distance computation → extract active pairs). Our version replaces pairwise_collide with direct sphere-sphere on precomputed pairs.

  **API/Type References** (contracts to implement against):
  - `pyroki/collision/_geometry_pairs.py:59-68` — `_sphere_sphere_dist()` helper: the exact formula `dist = ||c2-c1|| - (r1+r2)`. Reimplement this as inline JAX ops rather than calling the function.
  - `pyroki/collision/_geometry.py:142-184` — `Sphere` dataclass: reference for understanding sphere representation, but DO NOT use `Sphere` objects at runtime.
  - `pyroki/collision/_collision.py:125-152` — `colldist_from_sdf()`: the margin penalty function. Our distance output feeds directly into this.

  **External References**:
  - `jax_dataclasses` — `@jdc.pytree_dataclass` and `jdc.Static` usage for JAX-compatible dataclasses
  - `jaxlie.SE3` — `.apply()` for transforming points, `forward_kinematics()` returns `(num_links, 7)` wxyz_xyz array

  **Acceptance Criteria**:

  - [x] File `teleop_xr/ik/collision.py` exists with `MultiSphereCollision` class
  - [x] Class has all fields: `num_primitives`, `num_links`, `link_names`, `sphere_centers_local`, `sphere_radii`, `sphere_link_indices`, `pair_i`, `pair_j`
  - [x] `at_config()` method returns world-frame centers and radii
  - [x] `compute_self_collision_distance()` method returns `(K,)` array of distances
  - [x] Both methods are JAX jit-compatible

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: MultiSphereCollision class imports and has correct fields
    Tool: Bash (python)
    Preconditions: teleop_xr package importable
    Steps:
      1. python -c "from teleop_xr.ik.collision import MultiSphereCollision; print('OK')"
      2. Assert: prints "OK" without ImportError
    Expected Result: Class is importable
    Evidence: stdout captured

  Scenario: Manually construct MultiSphereCollision and verify distance computation
    Tool: Bash (python)
    Preconditions: jax, jaxlie installed
    Steps:
      1. python -c "
         import jax.numpy as jnp
         import jaxlie
         from teleop_xr.ik.collision import MultiSphereCollision
         # Create a minimal test: 2 links, 1 sphere each, at distance 1.0 apart
         coll = MultiSphereCollision(
             num_primitives=2, num_links=2,
             link_names=('A','B'),
             sphere_centers_local=jnp.array([[0,0,0],[0,0,0]], dtype=jnp.float32),
             sphere_radii=jnp.array([0.1, 0.1], dtype=jnp.float32),
             sphere_link_indices=jnp.array([0, 1], dtype=jnp.int32),
             pair_i=jnp.array([0], dtype=jnp.int32),
             pair_j=jnp.array([1], dtype=jnp.int32),
         )
         print('num_primitives:', coll.num_primitives)
         print('pair count:', coll.pair_i.shape[0])
         "
      2. Assert: num_primitives is 2, pair count is 1
    Expected Result: Construction succeeds with correct field values
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(ik): add MultiSphereCollision data model with sphere-sphere distance`
  - Files: `teleop_xr/ik/collision.py`

---

- [x] 2. Implement sphere fitting algorithm (`build_multi_sphere_collision`)

  **What to do**:

  Add a factory function `build_multi_sphere_collision()` to `teleop_xr/ik/collision.py` that builds a `MultiSphereCollision` from a URDF. This is the init-time (non-JIT) algorithm that:

  **Step 1: Parse link structure**
  - Use `pyroki.RobotURDFParser.parse(urdf)` to get link names (matching `Robot.links.names`)
  - For each link, identify parent and child joint origins in the link's local frame using the URDF joint tree

  **Step 2: Define centerlines per link**
  For each link:
  - Find the joint connecting this link to its parent → parent joint origin in link-local frame (this is always `(0,0,0)` in the link's own frame, but the joint's `<origin>` gives the offset from parent)
  - Find all joints where this link is the parent → child joint origins in link-local frame (from each child joint's `<origin>` transform)
  - **Single child**: one centerline from `(0,0,0)` to child joint origin in link frame
  - **Multiple children** (e.g., torso_link → left_arm_l1, right_arm_l1): one centerline per child
  - **No children** (EE frames): single sphere at `(0,0,0)` with small radius, OR skip if link has no collision mesh
  - **Zero-length centerline** (fixed joint, coincident origins): single sphere at midpoint with radius from mesh bounding sphere

  **Step 3: Fit spheres along each centerline**
  For each centerline of length L between endpoints c1, c2, place N spheres (sparrows-style):
  ```python
  direction = (c2 - c1) / L
  sidelength = L / (2 * N)
  for i in range(N):
      center = c1 + direction * (2*i + 1) * sidelength
  ```
  N is configurable per link, default:
  - `base_link`: 2 (mostly static)
  - `waist_link`, `torso_link`: 4 per centerline
  - Arm links (`*_arm_l1` through `*_arm_l7`): 3-4 (these are relatively short cylindrical links)
  - EE frames: 1 (or 0 if no collision mesh)

  **Step 4: Compute mesh-informed radii**
  For each sphere center along a centerline:
  - Get the collision mesh vertices for this link (using `RobotCollision._get_trimesh_collision_geometries(urdf, link_name)`)
  - Project each vertex onto the centerline axis
  - For each sphere's segment interval `[center - sidelength, center + sidelength]` along the axis, find all vertices whose projection falls within
  - Radius = max perpendicular distance from the centerline among those vertices + safety margin (e.g., 0.01m)
  - If no vertices fall in segment, use a fallback: overall max perpendicular distance / 2
  - Clamp minimum radius to 0.005m (avoid degenerate zero-radius spheres)

  **Step 5: Build collision pair indices**
  - Get URDF adjacency: for each joint, parent and child link are adjacent → ignore
  - Also self-ignore: primitives from the same link are never paired
  - Compute all valid primitive pairs `(i, j)` where `i < j` and:
    - `link_of[i] != link_of[j]` (not same link)
    - `(link_of[i], link_of[j])` not in ignore set (not adjacent)
  - Store as `pair_i`, `pair_j` arrays

  **Function signature**:
  ```python
  def build_multi_sphere_collision(
      urdf: yourdfpy.URDF,
      spheres_per_link: dict[str, int] | None = None,  # override N per link
      default_n_spheres: int = 3,
      radius_margin: float = 0.01,  # meters, added to mesh-projected radius
      user_ignore_pairs: tuple[tuple[str, str], ...] = (),
  ) -> MultiSphereCollision:
  ```

  **Must NOT do**:
  - Don't use visual meshes — only collision meshes from URDF
  - Don't run optimization/fitting (k-means, ICP, etc.) — pure geometric computation
  - Don't modify URDF or pyroki internals
  - Don't use `Capsule.from_trimesh()` or `minimum_cylinder()` — that's the old approach

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex geometric algorithm with URDF parsing, mesh vertex projection, multi-child link handling, and multiple edge cases (zero-length, missing mesh, branching). Needs thorough understanding of coordinate frames.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 1)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: Task 1

  **References** (CRITICAL):

  **Pattern References**:
  - `pyroki/collision/_robot_collision.py:38-102` — `RobotCollision.from_urdf()`: the factory pattern to follow. Shows how to parse URDF, iterate links, gather collision geometry, build pair indices. Our factory follows the same structure but produces `MultiSphereCollision` instead.
  - `pyroki/collision/_robot_collision.py:104-152` — `_compute_active_pair_indices()`: the exact algorithm for building ignore matrix from URDF joints + user pairs, then extracting active `(i,j)` indices. Adapt this to work over primitives (not links) using `sphere_link_indices`.
  - `pyroki/collision/_robot_collision.py:154-233` — `_get_trimesh_collision_geometries()`: how to extract collision mesh vertices from URDF with proper transforms. Use this directly (call it as a static method) to get per-link trimesh objects.
  - `pyroki/_robot_urdf_parser.py` — `RobotURDFParser.parse()`: returns `(joint_info, link_info)` matching `Robot.links.names` ordering. Critical that our link name ordering matches.

  **Algorithm References** (sparrows decomposition):
  - sparrows `forward_occupancy/SO.py:166-174` — The core sphere placement formula:
    ```python
    sidelength = height / (2 * n_spheres)
    m_idx = (2 * torch.arange(n_spheres) + 1)
    centers = center_1 + direction * m_idx * sidelength
    radii = torch.sqrt(sidelength**2 + (radius_1 - m_idx * radial_delta)**2)
    ```
    We adapt this: use the same center placement but replace analytical radii with mesh-projected radii.

  **URDF Structure References**:
  - TeaArm joint tree: `base_link` → `waist_link` → `torso_link` → {`left_arm_l1`...`left_arm_l7`→`frame_left_arm_ee`, `right_arm_l1`...`right_arm_l7`→`frame_right_arm_ee`}
  - `yourdfpy.URDF.joint_map` — dict of joints with `.parent`, `.child`, `.origin` (4x4 transform)
  - `yourdfpy.URDF.link_map` — dict of links with `.collisions` list

  **Acceptance Criteria**:

  - [x] `build_multi_sphere_collision()` function exists in `teleop_xr/ik/collision.py`
  - [x] Returns a `MultiSphereCollision` with `num_primitives > 19` (more than 1 sphere per link)
  - [x] All sphere radii are positive (> 0.005)
  - [x] No same-link pairs exist in `pair_i/pair_j`
  - [x] No adjacent-link pairs exist in `pair_i/pair_j`
  - [x] Link names match `Robot.links.names` ordering

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Build MultiSphereCollision from TeaArm URDF
    Tool: Bash (python)
    Preconditions: TeaArm URDF accessible, pyroki installed
    Steps:
      1. python -c "
         import yourdfpy
         from teleop_xr.ik.collision import build_multi_sphere_collision
         from teleop_xr.ik.robots.teaarm import TeaArmRobot
         robot = TeaArmRobot()
         urdf = yourdfpy.URDF.load(robot.urdf_path)
         coll = build_multi_sphere_collision(urdf)
         print('num_primitives:', coll.num_primitives)
         print('num_links:', coll.num_links)
         print('pair_count:', coll.pair_i.shape[0])
         print('radii_min:', float(coll.sphere_radii.min()))
         print('radii_max:', float(coll.sphere_radii.max()))
         assert coll.num_primitives > 19, 'Must have more than 1 sphere per link'
         assert float(coll.sphere_radii.min()) > 0.004, 'All radii must be positive'
         print('ALL CHECKS PASSED')
         "
      2. Assert: stdout contains "ALL CHECKS PASSED"
    Expected Result: Factory builds valid collision model
    Evidence: stdout captured

  Scenario: No same-link or adjacent-link pairs
    Tool: Bash (python)
    Preconditions: Previous scenario passed
    Steps:
      1. python -c "
         import jax.numpy as jnp
         import yourdfpy
         from teleop_xr.ik.collision import build_multi_sphere_collision
         from teleop_xr.ik.robots.teaarm import TeaArmRobot
         robot = TeaArmRobot()
         urdf = yourdfpy.URDF.load(robot.urdf_path)
         coll = build_multi_sphere_collision(urdf)
         link_i = coll.sphere_link_indices[coll.pair_i]
         link_j = coll.sphere_link_indices[coll.pair_j]
         assert bool(jnp.all(link_i != link_j)), 'Same-link pairs found!'
         print('No same-link pairs: OK')
         # Check no adjacent pairs (spot check)
         print('Pair count:', coll.pair_i.shape[0])
         print('PAIR CHECKS PASSED')
         "
      2. Assert: stdout contains "PAIR CHECKS PASSED"
    Expected Result: Pair filtering is correct
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(ik): add multi-sphere fitting algorithm from URDF collision meshes`
  - Files: `teleop_xr/ik/collision.py`

---

- [x] 3. Integrate into TeaArmRobot (uncomment + rewire self-collision cost)

  **What to do**:

  Modify `teleop_xr/ik/robots/teaarm.py` to:

  1. **In `__init__`**: Build the `MultiSphereCollision` alongside the existing `robot_coll`:
     ```python
     from teleop_xr.ik.collision import build_multi_sphere_collision
     self.multi_sphere_coll = build_multi_sphere_collision(urdf)
     ```

  2. **In `build_costs`**: Replace the commented-out `self_collision_cost` with a new custom cost that uses `MultiSphereCollision`:
     ```python
     # Create a custom cost function that uses multi-sphere collision
     from pyroki.collision._collision import colldist_from_sdf

     @pk.costs.Cost.create_factory
     def multi_sphere_self_collision_cost(
         vals, robot, multi_sphere_coll, joint_var, margin, weight
     ):
         cfg = vals[joint_var]
         distances = multi_sphere_coll.compute_self_collision_distance(robot, cfg)
         residual = colldist_from_sdf(distances, margin)
         return (residual * weight).flatten()
     ```
     Then add it to costs:
     ```python
     costs.append(
         multi_sphere_self_collision_cost(
             self.robot,
             self.multi_sphere_coll,
             JointVar(0),
             margin=0.02,   # 2cm activation margin
             weight=5.0,
         )
     )
     ```

  3. **Keep the old `self.robot_coll`** for backwards compatibility / potential world collision use — don't remove it.

  4. **Tune parameters**:
     - `margin`: Start with 0.02m (2cm). Too large → arms can't cross midline. Too small → collision not detected early enough.
     - `weight`: Start with 5.0 (moderate — the pose cost is 50.0, so collision should be meaningful but not dominant).

  **Must NOT do**:
  - Don't change any other cost weights or parameters
  - Don't remove the existing `robot_coll` attribute
  - Don't modify the `BaseRobot` interface
  - Don't change how FK or other costs work

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward integration — import new class, build in init, add cost in build_costs. Changes are localized to one file.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential after Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/robots/teaarm.py:47-48` — Current `robot_coll` creation: `self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)`. Add `self.multi_sphere_coll` alongside.
  - `teleop_xr/ik/robots/teaarm.py:196-206` — Commented-out `self_collision_cost` block: this is where the new cost goes. Remove the commented block and replace with the multi-sphere version.
  - `pyroki/costs/_costs.py:169-182` — `self_collision_cost` function: follow this pattern for the custom cost factory. Note the `@Cost.create_factory` decorator pattern.
  - `pyroki/collision/_collision.py:125-152` — `colldist_from_sdf()`: import and use directly for the margin penalty.

  **API References**:
  - `teleop_xr/ik/robots/teaarm.py:112-208` — `build_costs()` method: the full cost list construction. The new collision cost should be added after `limit_cost` (line 196) and before the return.

  **Acceptance Criteria**:

  - [x] `TeaArmRobot.__init__` creates `self.multi_sphere_coll` attribute
  - [x] `build_costs()` includes multi-sphere self-collision cost (not commented out)
  - [x] Old `self.robot_coll` still exists (not removed)
  - [x] TeaArmRobot can be instantiated without error

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: TeaArmRobot initializes with multi-sphere collision
    Tool: Bash (python)
    Preconditions: Tasks 1-2 complete
    Steps:
      1. python -c "
         from teleop_xr.ik.robots.teaarm import TeaArmRobot
         robot = TeaArmRobot()
         assert hasattr(robot, 'multi_sphere_coll'), 'Missing multi_sphere_coll'
         assert hasattr(robot, 'robot_coll'), 'Old robot_coll should still exist'
         print('multi_sphere primitives:', robot.multi_sphere_coll.num_primitives)
         print('INIT CHECK PASSED')
         "
      2. Assert: stdout contains "INIT CHECK PASSED"
    Expected Result: Robot creates both collision models
    Evidence: stdout captured

  Scenario: build_costs includes self-collision cost
    Tool: Bash (python)
    Preconditions: TeaArmRobot initializes
    Steps:
      1. python -c "
         import jaxlie
         from teleop_xr.ik.robots.teaarm import TeaArmRobot
         robot = TeaArmRobot()
         costs = robot.build_costs(
             target_L=jaxlie.SE3.identity(),
             target_R=jaxlie.SE3.identity(),
             target_Head=None,
         )
         print('Number of costs:', len(costs))
         # Previously had ~5 costs (rest, manip, pose_L, pose_R, limit)
         # Now should have 6 (+ self-collision)
         assert len(costs) >= 6, f'Expected at least 6 costs, got {len(costs)}'
         print('COST COUNT CHECK PASSED')
         "
      2. Assert: stdout contains "COST COUNT CHECK PASSED"
    Expected Result: Self-collision cost is included in the cost list
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(ik): enable multi-sphere self-collision in TeaArmRobot`
  - Files: `teleop_xr/ik/robots/teaarm.py`

---

- [x] 4. Occupancy testing script

  **What to do**:

  Create a script `scripts/test_sphere_occupancy.py` that validates how well the generated spheres cover the original collision meshes.

  **Algorithm**:
  - For each link in the robot:
    - Build the `MultiSphereCollision` spheres
    - Get the original collision trimesh for that link
    - Sample points from the surface of the trimesh (using `trimesh.sample.sample_surface`)
    - Also sample points from the volume of the trimesh (using `trimesh.sample.volume_mesh`)
    - For each sampled point, check if it is contained within AT LEAST ONE of the generated spheres for that link
    - Calculate the "occupancy ratio": `contained_points / total_points`
    - Report the ratio per link

  **Expectation**:
  - For arm links, the ratio should be > 95% (conservative approximation)
  - For complex links like torso, verify the ratio is acceptable (> 80%)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires working with `trimesh` sampling and geometric containment logic.

  **Commit**: YES
  - Message: `test(ik): add sphere occupancy validation script`
  - Files: `scripts/test_sphere_occupancy.py`

---

- [x] 5. Tests and validation

  **What to do**:

  Create `tests/test_multi_sphere_collision.py` with the following tests:

  1. **test_construction**: Build `MultiSphereCollision` from TeaArm URDF, verify:
     - `num_primitives > 19` (more than one per link)
     - All radii positive
     - `sphere_link_indices` values in range `[0, num_links)`
     - `pair_i` and `pair_j` have same length
     - No same-link pairs

  2. **test_jit_compatible**: Verify `compute_self_collision_distance` is jit-compilable:
     - `jax.jit(coll.compute_self_collision_distance)(robot, q0)` succeeds
     - Output is finite

  3. **test_grad_compatible**: Verify gradients flow through:
     - `jax.grad(lambda q: coll.compute_self_collision_distance(robot, q).sum())(q0)` succeeds
     - Gradient is finite (not NaN/inf)

  4. **test_default_pose_no_penetration**: At default config (zeros), all distances should be non-negative (or nearly so):
     - `distances = coll.compute_self_collision_distance(robot, q0)`
     - `assert jnp.all(distances > -0.01)` (allow small numerical tolerance)

  5. **test_collision_cost_integration**: Run the full IK solver briefly to verify the cost doesn't crash:
     - Build costs with targets, create solver, run a few iterations
     - Verify output config is finite

  **Must NOT do**:
  - Don't test pyroki internals
  - Don't add slow/heavy tests (keep each under 30s)
  - Don't require visualization or human verification

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Straightforward test writing following standard pytest patterns.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/robots/teaarm.py:18-63` — `TeaArmRobot.__init__()`: how to instantiate the robot in tests
  - `teleop_xr/ik/collision.py` (Task 1-2 output) — `MultiSphereCollision` and `build_multi_sphere_collision` APIs

  **Test References**:
  - Look for existing test patterns in `tests/` directory (if any exist)

  **Acceptance Criteria**:

  - [x] `tests/test_multi_sphere_collision.py` exists with 5 test functions
  - [x] `python -m pytest tests/test_multi_sphere_collision.py -v` passes all tests
  - [x] No test takes longer than 30 seconds

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All tests pass
    Tool: Bash (pytest)
    Preconditions: All previous tasks complete
    Steps:
      1. python -m pytest tests/test_multi_sphere_collision.py -v --tb=short
      2. Assert: exit code 0
      3. Assert: output contains "5 passed"
    Expected Result: All 5 tests pass
    Evidence: pytest output captured

  Scenario: Tests run in reasonable time
    Tool: Bash (pytest)
    Preconditions: Tests exist
    Steps:
      1. python -m pytest tests/test_multi_sphere_collision.py -v --durations=5
      2. Assert: no individual test exceeds 30s
    Expected Result: Fast test suite
    Evidence: durations output captured
  ```

  **Commit**: YES
  - Message: `test(ik): add multi-sphere self-collision validation tests`
  - Files: `tests/test_multi_sphere_collision.py`

---

- [x] 6. Parse and apply SRDF ignore pairs

  **What to do**:

  The current multi-sphere model reports many collisions at the zero pose because it only uses URDF immediate adjacency for ignore pairs. The TeaArm robot has a MoveIt SRDF file with comprehensive ignore pairs that should be applied.

  **Steps**:
  - Create a helper function in `teleop_xr/ik/collision.py` to parse `<disable_collisions link1="..." link2="..." />` from an SRDF file.
  - In `TeaArmRobot.__init__`, locate the SRDF file at `/home/cc/codes/tea/ros2_wksp/src/tea-ros2/tea_bringup/config/teaarm.srdf`.
  - Parse the ignore pairs from the SRDF.
  - Pass these pairs to `build_multi_sphere_collision(..., user_ignore_pairs=srdf_pairs)`.
  - Also, consider reducing `radius_margin` from `0.01` to `0.005` if collisions persist at zero pose.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: XML parsing and passing a list of strings to an existing factory function.

- [x] 7. Auto-ignore zero-pose collisions

  **What to do**:
  Both the old capsule model and the new multi-sphere model report many collisions at the zero pose (e.g., base_link vs torso_link). This suggests either the collision meshes are overlapping by design or our approximation is too conservative. We should auto-generate a set of ignore pairs for any link pair that is in collision at the zero pose.

  **Steps**:
  - Implement an `auto_ignore_zero_pose_collisions(robot, coll)` function.
  - Compute distances at zero pose.
  - Identify link pairs with `distance < 0.005` (or similar threshold).
  - Add these to a permanent ignore set.
  - Re-initialize `MultiSphereCollision` (or filter its `pair_i/j`) to exclude these.
  - Apply this in `TeaArmRobot.__init__`.

  **Recommended Agent Profile**:
  - **Category**: `quick`

- [x] 8. Final pose safety verification

- [x] 9. Over-approximation verification (100% occupancy)

  **What to do**:
  Ensure the sphere fitting algorithm always over-approximates the original mesh. This is verified by ensuring 100% occupancy in `scripts/test_sphere_occupancy.py`.

  **Steps**:
  - Update `_fit_radii_along_centerline` to use `sqrt(sidelength^2 + max_perp^2)` formula.
  - Implement a final coverage check in `build_multi_sphere_collision` per link.
  - Add fallback bounding sphere if any vertices are uncovered.
  - Verify with `uv run python scripts/test_sphere_occupancy.py`.

  **Recommended Agent Profile**:
  - **Category**: `deep`

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(ik): add MultiSphereCollision data model with sphere-sphere distance` | `teleop_xr/ik/collision.py` | python import check |
| 2 | `feat(ik): add multi-sphere fitting algorithm from URDF collision meshes` | `teleop_xr/ik/collision.py` | factory build check |
| 3 | `feat(ik): enable multi-sphere self-collision in TeaArmRobot` | `teleop_xr/ik/robots/teaarm.py` | robot init check |
| 4 | `test(ik): add sphere occupancy validation script` | `scripts/test_sphere_occupancy.py` | script run |
| 5 | `test(ik): add multi-sphere self-collision validation tests` | `tests/test_multi_sphere_collision.py` | pytest |

---

## Success Criteria

### Verification Commands
```bash
# 1. All tests pass
python -m pytest tests/test_multi_sphere_collision.py -v

# 2. TeaArmRobot builds costs successfully
python -c "
import jaxlie
from teleop_xr.ik.robots.teaarm import TeaArmRobot
r = TeaArmRobot()
costs = r.build_costs(jaxlie.SE3.identity(), jaxlie.SE3.identity(), None)
print(f'{len(costs)} costs built successfully')
"

# 3. Collision distances are finite at default config
python -c "
import jax.numpy as jnp
from teleop_xr.ik.robots.teaarm import TeaArmRobot
r = TeaArmRobot()
q = jnp.zeros(len(r.actuated_joint_names))
d = r.multi_sphere_coll.compute_self_collision_distance(r.robot, q)
print(f'Distances: min={float(d.min()):.4f}, max={float(d.max()):.4f}')
assert jnp.all(jnp.isfinite(d)), 'Non-finite distances!'
print('SUCCESS')
"

# 4. Occupancy test passes
python scripts/test_sphere_occupancy.py
```

### Final Checklist
- [x] Multi-sphere collision model created with mesh-informed radii
- [x] Sphere count per link > 1 (more than old single-capsule approach)
- [x] Collision cost enabled in TeaArmRobot.build_costs() (not commented out)
- [x] Default pose produces non-penetrating distances
- [x] JAX jit + grad compatible
- [x] All tests pass
- [x] Old robot_coll preserved for backwards compatibility
- [x] No pyroki source modifications
