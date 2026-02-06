# Plan: Implement Collision Ignore Pair Generation

## Context
The user wants to automate the generation of collision ignore pairs for the TeaArm robot. Currently, `scripts/generate_spheres.py` generates a `sphere.json` containing only sphere data. The goal is to:
1.  Enhance `scripts/generate_spheres.py` to calculate collision ignore pairs by sampling random configurations and identifying pairs that are *always* colliding (structural) or *never* colliding (distant).
2.  Update the output format to `collision.json` which includes both spheres and ignore pairs.
3.  Update `teleop_xr/ik/robots/teaarm.py` to consume this new format.

This will improve collision checking performance and accuracy by excluding irrelevant pairs.

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Modifies the generation script (independent) |
| Task 2 | Task 1 | Modifies the consumer to match the generator's format (logically dependent) |

## Parallel Execution Graph

Wave 1 (Start immediately):
├── Task 1: Update `scripts/generate_spheres.py` (Add generation logic)
└── Task 2: Update `teleop_xr/ik/robots/teaarm.py` (Update loader logic)

Critical Path: Task 1 → Task 2 (Logical flow, though code can be written in parallel)

## Tasks

### Task 1: Update Sphere Generation Script
**Description**:
Modify `scripts/generate_spheres.py` to:
1.  Implement a `compute_collision_ignore_pairs(robot, n_samples=1000)` function.
    *   Sample `n_samples` random configurations within joint limits.
    *   Compute mesh distances using `robot.get_mesh_distances(cfg)`.
    *   Identify pairs that are **always** in collision (count == n_samples) OR **never** in collision (count == 0).
    *   Return these as a list of lists/tuples.
2.  Update `default_export_path` to use `collision.json`.
3.  Update `on_export` callback to:
    *   Call `compute_collision_ignore_pairs`.
    *   Structure output as `{"spheres": {...}, "collision_ignore_pairs": [...]}`.
    *   Write to `collision.json`.

**Delegation Recommendation**:
- Category: `visual-engineering` - Script involves 3D geometry and existing Viser GUI.
- Skills: [`brainstorming`, `simplify`] - Logic implementation.

**Skills Evaluation**:
- INCLUDED `brainstorming`: To ensure edge cases in collision logic are handled.
- OMITTED `frontend-ui-ux`: No UI changes needed, just backend logic hooked to button.

**Depends On**: None

**Acceptance Criteria**:
- [ ] `compute_collision_ignore_pairs` function exists and runs without error.
- [ ] Export button produces `collision.json` with `spheres` and `collision_ignore_pairs` keys.
- [ ] `collision_ignore_pairs` contains valid link name pairs.
- [ ] Default filename in GUI is `collision.json`.

### Task 2: Update TeaArm Robot Loader
**Description**:
Modify `teleop_xr/ik/robots/teaarm.py` to support the new collision format while maintaining backward compatibility.
1.  Update `_load_sphere_decomposition`:
    *   Check for `collision.json` first.
    *   Fallback to `sphere.json` if not found.
2.  Update `__init__`:
    *   Handle new dictionary structure (`{"spheres": ..., "collision_ignore_pairs": ...}`).
    *   Handle old dictionary structure (direct sphere dict).
    *   Pass `user_ignore_pairs` (as tuple of tuples) to `pk.collision.RobotCollision.from_sphere_decomposition`.

**Delegation Recommendation**:
- Category: `unspecified-high` - Python backend logic.
- Skills: [`simplify`] - Clean logic for format detection.

**Skills Evaluation**:
- INCLUDED `simplify`: Keep the loading logic clean and readable.

**Depends On**: Task 1 (Format definition)

**Acceptance Criteria**:
- [ ] `TeaArmRobot` initializes successfully with new `collision.json` format.
- [ ] `TeaArmRobot` initializes successfully with old `sphere.json` format (if `collision.json` missing).
- [ ] Ignore pairs are correctly passed to `pyroki`.

## Commit Strategy
- Commit 1: "feat(ik): add collision ignore pair generation to scripts/generate_spheres.py"
- Commit 2: "feat(ik): update TeaArmRobot to load collision.json with ignore pairs"

## Success Criteria
- `scripts/generate_spheres.py` runs and exports `collision.json`.
- `collision.json` contains non-empty `collision_ignore_pairs`.
- `TeaArmRobot` loads without errors using the new file.
