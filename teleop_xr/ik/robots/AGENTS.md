# teleop_xr/ik/robots/ — Robot Definitions

Pluggable robot models with kinematic chains and cost functions.

## OVERVIEW
Each robot inherits `BaseRobot` and defines its URDF, joints, and IK costs. Assets (collision spheres, custom meshes) stored in `assets/` subdirectory.

## STRUCTURE
```
robots/
├── franka.py         # Franka Panda (7-DOF arm)
├── h1_2.py           # Unitree H1 humanoid
├── openarm.py        # OpenArm
└── assets/           # Robot-specific assets
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new robot | `<name>.py` | Copy franka.py as template |
| Custom collision | `assets/<robot>/collision.json` | Sphere decomposition |
| Entry points | `../../pyproject.toml` | `[project.entry-points."teleop_xr.robots"]` |

## CONVENTIONS

### Robot Loading
- **Standard**: URDF via RAM (`ram.get_resource("franka/panda.urdf")`)
- **Custom**: Local assets in `assets/<robot>/`

### Cost Tuning
- Pose weight: 1e3 (default)
- Rest weight: 1e0 (regularization)
- Adjust in `build_costs` method

## ANTI-PATTERNS (robots)

- ❌ NEVER hardcode URDFs (use RAM)
- ❌ NEVER call remote repos during collision generation
- ✅ ALWAYS test FK before IK

## NOTES

**Outlier**: `teleop_xr/utils/lite6.urdf` should move here
