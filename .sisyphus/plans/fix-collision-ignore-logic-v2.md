# Fix Collision Ignore Logic - Round 2

## TL;DR

> **Quick Summary**: Ensure End-Effector (EE) vs Torso collisions are consistently NOT ignored, while clearing large structural overlaps at the home pose. Change the policy to use a higher adjacency order (N=4) and re-enable a very tight (1mm) zero-pose auto-ignore that EXEMPTS End-Effector links.
>
> **Deliverables**:
> - Updated `TeaArmRobot` to use `ignore_adj_order=4`.
> - Re-implemented zero-pose auto-ignore with **1mm** threshold and **EE Exemption**.
> - Updated tests to reflect the new policy.

---

## Context

### Problem
- The previous "symmetrize" fix auto-ignored BOTH left and right EE vs Torso because the 5mm threshold was too loose (Right EE was -16mm at home).
- Structural overlaps (L3/L4 vs Torso) are as high as -178mm, making N=2 adjacency insufficient.
- User wants EE vs Torso active, but wants to use "adjacent link auto ignore" for the rest.

### Strategy
1. **Adjacency First**: Use `ignore_adj_order=4` to clear the proximal arm overlaps.
2. **Tight Auto-Ignore**: Re-enable distance-based ignore at home pose, but with a **1mm** threshold (much tighter than 5mm or the misidentified 50mm).
3. **EE Exemption**: Explicitly FORBID auto-ignoring any link pair involving an End-Effector against the Torso, regardless of distance.

---

## TODOs

- [x] 1. Update ignore logic in `collision.py` and `teaarm.py`

  **What to do**:

  1. In `teleop_xr/ik/robots/teaarm.py`:
     - Set `ignore_adj_order=4` in `build_multi_sphere_collision`.
     - Re-implement a tight `_collect_zero_pose_ignore_pairs` with `distance_threshold=0.001` (1mm).
     - Add a check inside the collection loop: **Skip (do NOT ignore)** if the pair is `(torso_link, *ee*)`.
  2. Verify that `base_link` vs `left_arm_l7` (EE) is ignored if it's < 1mm (it's currently -96mm, so it will be ignored if N=4 doesn't catch it).

- [x] 2. Update Tests
- [x] 3. Fix EE exemption to include l7 links

  **What to do**:

  The physical links that should be active for end-effector collisions are the `l7` links. The current naming check `is_ee` misses these.

  **Steps**:
  - Update `_collect_zero_pose_ignore_pairs` in `teleop_xr/ik/robots/teaarm.py`.
  - Add `link.endswith("_l7")` to the `is_ee_i` and `is_ee_j` checks.
  - Verification: `frame_left_arm_ee`, `left_arm_l7`, `frame_right_arm_ee`, and `right_arm_l7` are all protected from auto-ignore when paired with `torso_link`.

- [x] 4. Final verification of all EE collisions

  **What to do**:
  - Update `tests/test_multi_sphere_collision.py` to assert that `frame_left_arm_ee` vs `torso_link` exists in the primitive pair set.
  - Update `test_default_pose_no_penetration` to use the new 1mm tolerance.
