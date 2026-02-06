### Findings - 2026-02-06
- Updated TeaArmRobot to use ignore_adj_order=4.
- Implemented _collect_zero_pose_ignore_pairs with 1mm threshold.
- End-Effector links (frame_* or *ee*) are exempted from auto-ignore when colliding with torso_link.
- Note: frame_left_arm_ee and frame_right_arm_ee currently have 0 spheres because they lack collision geometry in the URDF, so they are naturally absent from active collision pairs.
- Observed that right_arm_l7 collides with torso_link by -16mm at home. Since it doesn't match the EE name criteria, it is auto-ignored.

### Findings - 2026-02-06 (Update)
- Updated tests/test_multi_sphere_collision.py:
    - Added test_ee_torso_collision_not_ignored to verify left_arm_l7 vs torso_link is active.
    - Updated test_default_pose_no_penetration tolerance to -0.001 (1mm).
- Confirmed that left_arm_l7 vs torso_link is NOT ignored because its zero-pose distance (33mm) is above the 1mm auto-ignore threshold.
- Confirmed that right_arm_l7 vs torso_link IS ignored because its zero-pose distance (-18mm) is below the 1mm threshold and it is not yet recognized as an EE link in teaarm.py.
### EE Exemption Logic Update
- Added `link.endswith("_l7")` to the EE exemption logic in `TeaArmRobot._collect_zero_pose_ignore_pairs`.
- This ensures that links like `right_arm_l7` are treated as EE links and thus NOT auto-ignored when colliding with the `torso_link` at zero pose.
- Verification script confirmed that `('right_arm_l7', 'torso_link')` is no longer in the auto-ignore list.
