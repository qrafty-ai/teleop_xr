Updated _default_spheres_for_link to handle both left and right arms identically by checking for both '_arm_l' and '_arm_r' in the link name. This ensures that the collision spheres are correctly assigned for right-arm links as well.

In `_fit_radii_along_centerline`, using `max_perp = 0.0` for empty centerline segments keeps fallback spheres tight (`sqrt(sidelength^2 + 0)` plus margin) and avoids propagating the widest cross-section radius into narrow/tapered regions.

Replacing origin fallback with iterative centroid patch spheres works best when coverage checks include richer mesh points (vertices + edge midpoints + face centers + deterministic sampled surface/volume points); this catches uncovered lobes without inflating proximal arm spheres.

Constraining patch-sphere radius to 0.15 m on arm links keeps L1/L2 radii compact (~0.077 m / ~0.044 m max) while still achieving full occupancy in the verification script.

Ancestor/descendant-only N-hop ignore can be implemented by walking each link up its parent chain for at most N hops using URDF joint parent->child relations; this avoids accidentally ignoring sibling links on branching subtrees.

For N > 1, keep `torso_link` vs end-effector pairs active by exempting links whose names contain `ee` or start with `frame_`; this preserves critical EE-vs-torso collision checks even when proximal torso-arm pairs are ignored.

Synthetic URDF tests with explicit link names (`left_arm_l1`, `left_arm_l2`, `frame_left_arm_ee`, `torso_link`) are sufficient to validate ignore-set behavior deterministically without depending on external robot resources.

- Removed zero-pose auto-ignore logic from TeaArmRobot as it was too loose and caused distal collisions to be ignored.
- Transitioned to using N-hop ignore logic (ignore_adj_order=2) in build_multi_sphere_collision, which provides a more robust and predictable set of ignored pairs based on robot topology.
- Verified that TeaArmRobot still achieves 100% sphere occupancy for arm links with the new configuration.
