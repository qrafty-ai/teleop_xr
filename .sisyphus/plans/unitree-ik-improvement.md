# Unitree IK Improvement Plan

## TL;DR

> **Quick Summary**: Improve Unitree H1 IK stability by adding Damping (anti-sudden motion), Manipulability (anti-singularity), and Self-Collision avoidance.
>
> **Deliverables**:
> - Updated `BaseRobot` interface (accepts `q_current`)
> - Updated `PyrokiSolver` to pass `q_current`
> - Updated `UnitreeH1Robot` with 3 new costs
> - Updated `FrankaRobot` to match interface
>
> **Estimated Effort**: Short
> **Parallel Execution**: Sequential (due to interface dependency)
> **Critical Path**: BaseRobot Interface → Solver Update → Robot Implementations

---

## Context

### Original Request
Improve Unitree robot IK to:
1. Prevent sudden large motion
2. Prevent singularity
3. (Optional) Self collision avoidance

### Interview Summary
**Key Discussions**:
- Confirmed Robot: Unitree H1 (`h1_2.py`)
- Confirmed Lib: `pyroki` (has built-in costs)
- Technical Strategy:
    - **Sudden Motion**: `rest_cost` targeting `q_current` (Damped Least Squares effect)
    - **Singularity**: `manipulability_cost` (Active Avoidance)
    - **Collision**: `self_collision_cost`

### Metis Review (Self-Simulated)
**Identified Gaps** (addressed):
- **Interface Breakage**: `FrankaRobot` inherits from `BaseRobot`, must be updated to avoid crash.
- **Weights**: Need tunable defaults. Damping=1.0, Manip=1.0, Collision=100.0 (high priority).

---

## Work Objectives

### Core Objective
Implement robust IK costs for Unitree H1 to ensure safety and stability.

### Concrete Deliverables
- [x] `teleop_xr/ik/robot.py`: Updated `build_costs` signature
- [x] `teleop_xr/ik/solver.py`: Pass `q_current` to `build_costs`
- [x] `teleop_xr/ik/robots/h1_2.py`: New costs implementation
- [x] `teleop_xr/ik/robots/franka.py`: Interface compatibility fix

### Definition of Done
- [x] `uv run python -m teleop_xr.demo --mode ik` runs without error
- [x] H1 robot avoids self-collision in demo
- [x] Motion is smooth (no instant jumps)

### Must Have
- Damping cost (stabilization)
- Manipulability cost (singularity avoidance)
- Interface compatibility for all robots

### Must NOT Have (Guardrails)
- New external dependencies (use existing `pyroki`)
- Changes to `forward_kinematics` logic

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> All verification is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: YES (pytest) but mostly for unit logic.
- **Automated tests**: NO (Logic is visual/physics-based).
- **Strategy**: **Agent-Executed QA Scenarios** using `demo` script.

### Agent-Executed QA Scenarios

```
Scenario: IK Demo Loads and Solves
  Tool: interactive_bash (tmux)
  Preconditions: Virtual environment ready
  Steps:
    1. tmux new-session: uv run python -m teleop_xr.demo --mode ik
    2. Wait for: "TeleopXR" in output (timeout: 10s)
    3. Assert: No "TypeError" or "NotImplementedError" in stderr
    4. Assert: "FPS" is updating (loop is running)
    5. Send keys: "q"
    6. Assert: Exit code 0
  Expected Result: Demo runs with new IK logic without crashing
  Evidence: Terminal output captured
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Sequential Chain):
└── Task 1: Refactor Interface & Update Solvers (BaseRobot + Solver + Franka)
    └── Task 2: Implement Unitree H1 Improvements (The core logic)
```

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | quick (refactor) |
| 2 | 2 | visual-engineering (robot logic) |

---

## TODOs

- [x] 1. Refactor BaseRobot Interface & Update Dependents

  **What to do**:
  - Update `teleop_xr/ik/robot.py`: `BaseRobot.build_costs` adds `q_current: jnp.ndarray | None = None` argument.
  - Update `teleop_xr/ik/solver.py`: `_solve_internal` passes `q_current` to `build_costs`.
  - Update `teleop_xr/ik/robots/franka.py`: Update `build_costs` signature to match BaseRobot (ignore `q_current` for now).

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`simplify`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 2

  **References**:
  - `teleop_xr/ik/robot.py:86` - Base signature
  - `teleop_xr/ik/solver.py:59` - Call site
  - `teleop_xr/ik/robots/franka.py` - Dependent subclass

  **Acceptance Criteria**:
  - [ ] `uv run python -c "from teleop_xr.ik.robot import BaseRobot; print(BaseRobot.build_costs.__annotations__)"` shows `q_current`
  - [ ] `uv run python -m teleop_xr.demo --mode ik` starts (even if H1 ignores new arg for now)

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Verify Interface Compatibility
    Tool: Bash
    Preconditions: None
    Steps:
      1. Run: uv run python -c "from teleop_xr.ik.robots.franka import FrankaRobot; r=FrankaRobot(); print('Franka instantiated')"
      2. Assert: Output contains "Franka instantiated"
    Expected Result: No abstract method instantiation errors
  ```

- [x] 2. Implement Unitree H1 IK Improvements

  **What to do**:
  - Modify `teleop_xr/ik/robots/h1_2.py`:
  - Update `build_costs` signature.
  - Add `pk.costs.rest_cost(..., target=q_current, weight=1.0)` (Damping).
  - Add `pk.costs.manipulability_cost(..., weight=1.0)` (Singularity).
  - Add `pk.costs.self_collision_cost(..., weight=100.0)` (Collision).
  - Ensure weights are float/array compatible with JAX.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`cartography`] (for understanding robot structure)

  **Parallelization**:
  - **Blocked By**: Task 1

  **References**:
  - `teleop_xr/ik/robots/h1_2.py` - File to modify
  - `teleop_xr/ik/solver.py` - Context for how costs are used

  **Acceptance Criteria**:
  - [ ] `h1_2.py` includes `rest_cost`, `manipulability_cost`, `self_collision_cost`
  - [ ] `uv run python -m teleop_xr.demo --mode ik` runs smoothly
  - [ ] No "nan" or infinite values in solver output (implicit check via demo stability)

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Unitree H1 IK Stability Test
    Tool: interactive_bash (tmux)
    Preconditions: None
    Steps:
      1. tmux new-session: uv run python -m teleop_xr.demo --mode ik
      2. Wait for: "FPS" (system running)
      3. Sleep: 5s (allow solver to run)
      4. Send keys: "q"
      5. Assert: Exit code 0
    Expected Result: Solver runs stable with new costs
  ```

  **Commit**: YES
  - Message: `feat(ik): add damping, singularity and collision avoidance to Unitree H1`
  - Files: `teleop_xr/ik/robots/h1_2.py`, `teleop_xr/ik/robot.py`, `teleop_xr/ik/solver.py`

---

## Success Criteria

### Final Checklist
- [x] Interface updated across all 3 files (Base, Solver, Franka)
- [x] Unitree H1 has 3 new active costs
- [x] Demo runs without crashing
