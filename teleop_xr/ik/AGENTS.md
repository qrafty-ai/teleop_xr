# teleop_xr/ik/ — Inverse Kinematics System

JAX-powered IK solver using pyroki and jaxls for real-time robot control.

## OVERVIEW
Translates XR controller poses into robot joint configurations via relative-motion IK. Supports multiple robots (Franka, H1, OpenArm) with pluggable cost functions.

## STRUCTURE
```
ik/
├── robot.py          # BaseRobot interface
├── solver.py         # PyrokiSolver (JAX-compiled)
├── controller.py     # IKController (engagement, snapshots)
├── loader.py         # Dynamic robot loading
└── robots/           # Robot implementations
    ├── franka.py
    ├── h1_2.py
    ├── openarm.py
    └── assets/       # Robot-specific configs
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Define new robot | `robot.py` | Inherit BaseRobot, implement FK + costs |
| IK solver loop | `solver.py` | JAX JIT-compiled, ~15 iterations/frame |
| Relative motion | `controller.py` | `Target = Init + (XR_now - XR_init)` |
| Robot loading | `loader.py` | Entry points + module paths |

## CONVENTIONS

### Robot Definition
- Subclass `BaseRobot`
- Implement: `forward_kinematics`, `build_costs`, `actuated_joint_names`
- Optional: `get_vis_config` for frontend URDF

### Cost Functions
- **Pose**: Minimize end-effector error
- **Rest**: Regularization (home position)
- **Limits**: Joint bounds penalties
- **Manipulability**: Avoid singularities

### Deadman Rule
- Both grips must be squeezed for IK engagement
- Single-arm operation forbidden (safety)

## ANTI-PATTERNS (ik)

- ❌ NEVER change solver algorithm without failing evidence
- ❌ NEVER weaken geometry assertions to hide collision issues
- ✅ ALWAYS snapshot robot state on engagement

## NOTES

### Performance
- JAX JIT compilation on first run
- Pyroki for differentiable kinematics
- Filter outputs with `WeightedMovingFilter`
