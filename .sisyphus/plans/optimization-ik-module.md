# Optimization IK Module for Teleop XR

## TL;DR

> **Quick Summary**: Create a modular `teleop_xr/ik` package using Pyroki (JAX) for real-time bimanual IK on the Unitree H1_2 robot. The system uses a deadman switch to gate control, calculates relative pose deltas using a **decoupled rotation/translation algorithm** (code provided), and uses a **robot-specific** optimization formulation to map these targets to the robot.
>
> **Deliverables**:
> - `teleop_xr/ik/` module with `BaseRobot`, `UnitreeH1Robot`, and `PyrokiSolver`
> - **Copied Assets**: `assets/h1_2/` copied into project.
> - `teleop_xr/ik/controller.py` with correct delta pose logic.
> - `teleop_xr.demo_ik` entrypoint & tests.
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves

---

## Technical Approach

### 1. Delta Pose Logic (Reference Implementation)
The controller MUST use this exact implementation for calculating target poses from controller inputs. Do NOT deviate from this logic.

```python
from spatialmath import SE3, UnitQuaternion

def compute_teleop_transform(
    self, t_ctrl_curr: SE3, t_ctrl_init: SE3, t_ee_init: SE3
) -> SE3:
    """Calculate the new EE pose by applying the controller's relative motion independently to rotation and translation.

    Args:
        t_ctrl_curr: Current controller transform
        t_ctrl_init: Initial controller transform
        t_ee_init: Initial EE transform
    Returns:
        New EE transform

    """
    # --- Rotation (Using UnitQuaternion) ---
    # 1. Convert SE3 rotations to Quaternions
    q_ctrl_curr = UnitQuaternion(t_ctrl_curr)
    q_ctrl_init = UnitQuaternion(t_ctrl_init)
    q_ee_init = UnitQuaternion(t_ee_init)

    # 2. Calculate the global rotation delta of the controller
    # delta = Current * Init_Inverse
    q_delta = q_ctrl_curr * q_ctrl_init.inv()

    # 3. Apply this delta to the initial EE rotation
    q_new = q_delta * q_ee_init

    # --- Translation (Vector Arithmetic) ---
    # 1. Calculate translation delta
    t_delta = t_ctrl_curr.t - t_ctrl_init.t

    # 2. Apply delta to initial EE position
    t_new = t_ee_init.t + t_delta

    # --- Recombine into SE3 ---
    # SE3.Rt(Rotation Matrix, Translation Vector)
    return SE3.Rt(q_new.R, t_new)
```

### 2. Robot-Specific Cost Formulation
The `PyrokiSolver` will be designed to accept a cost factory or strategy from the `Robot` class, allowing `UnitreeH1Robot` to define its own unique cost structure (including the "Yaw Trick" for the waist).

**Unitree H1_2 Costs:**
- **Arm Tracking**: `pose_cost` for L/R end-effectors.
- **Waist Yaw**: Custom cost aligning head yaw to waist yaw.
- **Regularization**: `rest_cost` (weighted heavily for legs to freeze them) + `limit_cost`.

### 3. Asset Management
- **Action**: Copy `h1_2.urdf` and associated meshes from `~/.cache/...` to `teleop_xr/assets/h1_2/`.
- **Benefit**: Self-contained project, consistent paths.

### 4. Utilities
- **Moving Filter**: Copy `WeightedMovingFilter` from `openarm_teleop_node.py` (or similar reference) into `teleop_xr/utils/` since it doesn't exist in the base repo yet.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation):
├── Task 1: Install Dependencies & Copy Assets
├── Task 2: Create IK package & BaseRobot class
└── Task 3: Implement Utilities (WeightedMovingFilter)

Wave 2 (Implementation):
├── Task 4: Implement UnitreeH1Robot (with cost definition)
├── Task 5: Implement PyrokiSolver (accepting robot costs)
└── Task 6: Implement IKController (with decoupled delta logic)

Wave 3 (Integration):
└── Task 7: Demo & Integration
```

---

## TODOs

- [x] 1. Install Dependencies & Copy Assets
- [x] 2. Create IK package and BaseRobot class
- [x] 3. Implement Utilities
- [x] 4. Implement UnitreeH1Robot
- [x] 5. Implement PyrokiSolver
- [x] 6. Implement IKController
- [x] 7. Demo Entrypoint


  **What to do**:
  - Create `teleop_xr/demo_ik.py`.
  - Wire up everything and run.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`python`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocked By**: Task 6
