# Fix Collision Ignore Logic and Sphere Fitting

## TL;DR

> **Quick Summary**: Fix the issue where arm-vs-torso collisions were being auto-ignored by removing the unstable zero-pose distance-based ignore logic. Replace it with a robust N-th order adjacency policy (ancestor/descendant only). Simultaneously fix the sphere fitting algorithm to prevent "exploding" fallback spheres that caused those overlaps in the first place.
>
> **Deliverables**:
> - Removed `_collect_zero_pose_ignore_pairs` and associated logic in `TeaArmRobot`.
> - Improved `_fit_radii_along_centerline` to use local-max radii instead of global-max for empty segments.
> - Replaced origin-centered fallback spheres with centroid-centered "patch spheres" for 100% coverage.
> - New `nth_order_adjacency` ignore logic in `build_multi_sphere_collision`.
> - Updated tests ensuring EE vs Torso is NOT ignored.

---

## Context

### Problem
The current implementation was auto-ignoring all arm links against the torso because my overly conservative sphere fitting (100% over-approximation) caused "collisions" at the zero pose. This resulted in zero collision detection for the hands vs the body.

### Strategy
1. **Remove Unstable Logic**: Delete the zero-pose auto-ignore. It's too sensitive to approximation errors.
2. **Fix Fitting**: Ensure spheres stay tight to the mesh. Use local fallback clusters instead of the link origin for missed vertices.
3. **Robust Adjacency**: Implement N-hop ignore (ancestor/descendant only, N=2) to allow legitimate proximal overlaps (shoulder) while keeping distal ones (hands) active.

---

## Work Objectives

### Core Objective
Ensure high-fidelity self-collision detection where arm end-effectors are consistently checked against the torso, while proximal links (shoulder) are appropriately ignored using a deterministic adjacency policy.

### Concrete Deliverables
- `teleop_xr/ik/collision.py` — Updated `build_multi_sphere_collision` with N-hop ignore.
- `teleop_xr/ik/collision.py` — Fixed `_fit_radii_along_centerline` and local patch sphere logic.
- `teleop_xr/ik/robots/teaarm.py` — Removed zero-pose auto-ignore logic.
- `tests/test_collision_fix.py` — New verification tests.

### Definition of Done
- [ ] `TeaArmRobot` no longer uses zero-pose auto-ignore.
- [ ] `frame_left_arm_ee` vs `torso_link` is NOT ignored (appears in `pair_i/pair_j`).
- [ ] No sphere radius exceeds 0.20m for arm links.
- [ ] All 100% occupancy requirements are still met.
- [ ] IK solver is stable at zero pose.

---

## Verification Strategy

### QA Scenarios

- **Scenario 1**: Verify `EE vs Torso` pair existence in final model.
- **Scenario 2**: Verify 100% occupancy with the improved fitting.
- **Scenario 3**: Verify no "beach ball" spheres (max radius check).

---

## TODOs

- [x] 1.1 Symmetrize default spheres for _arm_r
- [x] 1.2 Refine _fit_radii_along_centerline empty fallback
- [x] 1.3 Replace origin-fallback with local patch spheres

- [x] 2. Implement N-th order adjacency ignore logic

  **What to do**:

  Modify `build_multi_sphere_collision` in `teleop_xr/ik/collision.py`:
  1. Calculate the kinematic adjacency matrix (ancestor/descendant only).
  2. For each link pair, ignore if graph distance (hops) <= N (default N=1).
  3. For TeaArm, use N=2 to ignore `torso` vs `l1` and `torso` vs `l2`.
  4. Ensure `EE` links are EXEMPT from N-hop ignoring against the torso.

- [ ] 3. Cleanup TeaArmRobot and verify

  **What to do**:

  Modify `teleop_xr/ik/robots/teaarm.py`:
  1. Delete `_collect_zero_pose_ignore_pairs`.
  2. Simplify `__init__`: only parse SRDF and build final collision model.
  3. Remove the "temporary model" build step.
  4. Run `scripts/test_sphere_occupancy.py` and a new test script to verify EE vs Torso is active.

- [ ] 4. Final Validation

  **What to do**:
  - Run all tests.
  - Verify zero collisions at home pose (if any remain, add them to SRDF or adjust N).
