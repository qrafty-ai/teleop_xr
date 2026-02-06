# [SUPERSEDED] Replace Self-Collision with Dynamic Sphere Collision (Ballpark-First)

This plan has been superseded by the static sphere loading approach in `.sisyphus/plans/static-sphere-collision.md`.
All relevant features have been migrated and dynamic calculation code removed from the main package.


## TL;DR

> **Quick Summary**: Replace the current URDF capsule self-collision path with runtime-generated sphere decompositions built via a ballpark-based pipeline, then enforce over-approx safety with validation/fallback before wiring into Pyroki's sphere collision API for `h1_2` and `teaarm`.
>
> **Deliverables**:
> - Dynamic sphere decomposition generator (strict over-approximation)
> - Deterministic cache with lock-safe read/write
> - User-visible approximation status messages (cache miss/generation/hit/invalidation)
> - `h1_2` and `teaarm` migrated to `RobotCollision.from_sphere_decomposition(...)`
> - Re-enabled self-collision cost in both migrated robots
> - TDD coverage for schema, over-approximation, cache behavior, and solver path
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 2 parallel waves in the middle
> **Critical Path**: Task 1 -> Task 2 -> Task 4 -> Task 5 -> Task 6

---

## Context

### Original Request
Plan migration from self-collision checking to sphere collision checking using the latest Pyroki support, with dynamic sphere generation and mesh over-approximation guarantees.

### Interview Summary
**Key Discussions**:
- Scope is explicitly limited to `teleop_xr/ik/robots/h1_2.py` and `teleop_xr/ik/robots/teaarm.py`.
- `franka` is explicitly out of scope for this pass.
- Sphere decomposition must be generated dynamically (not precommitted static JSON assets).
- SPARROWS approach was reviewed and rejected for this codebase.
- Chosen method is a **ballpark-based sphere generation pipeline**.
- Test mode is **TDD**.
- User confirmed the previously wrong Pyroki lock was fixed.

**Research Findings**:
- Existing self-collision hooks are currently commented out in:
  - `teleop_xr/ik/robots/h1_2.py`
  - `teleop_xr/ik/robots/teaarm.py`
- Project test infrastructure exists and is active:
  - Python `pytest` via `pyproject.toml`
  - Existing robot and solver tests in `tests/`
- Pyroki sphere decomposition contract supports runtime dictionary input:
  - `{"link_name": {"centers": [[x,y,z], ...], "radii": [r, ...]}}`
- RAM cache and lock patterns already exist and are reusable.

### Metis Review
**Identified Gaps (resolved in this plan)**:
- Frame ambiguity resolved by adding explicit transform-chain implementation tasks and containment tests in the same frame used by Pyroki.
- Determinism risk resolved by pinning decomposition parameters and deterministic ordering rules.
- Runtime cost risk resolved by mandatory cache task with cache-hit verification and fallback behavior.
- Verification gap resolved by requiring strict over-approximation tests (not only schema checks).
- Scope creep risk resolved by explicit guardrails excluding `franka` and non-collision semantic changes.

---

## Work Objectives

### Core Objective
Switch IK self-collision construction for `h1_2` and `teaarm` from URDF capsule fitting to runtime sphere decomposition generated from link collision geometry using a ballpark-based method, while preserving solver behavior and adding robust safety checks.

### Concrete Deliverables
- New decomposition module: `teleop_xr/ik/collision_spheres.py`
- New cache module: `teleop_xr/ik/collision_sphere_cache.py`
- Robot integration updates:
  - `teleop_xr/ik/robots/h1_2.py`
  - `teleop_xr/ik/robots/teaarm.py`
- Dependency/config updates for decomposition backend:
  - `pyproject.toml`
- New/updated tests:
  - `tests/test_collision_sphere_contract.py`
  - `tests/test_collision_sphere_generation.py`
  - `tests/test_collision_sphere_cache.py`
  - updates in `tests/test_robots.py`
  - updates in `tests/test_solver_optional.py`

### Definition of Done
- [x] `uv run pytest tests/test_collision_sphere_contract.py` passes
- [x] `uv run pytest tests/test_collision_sphere_generation.py` passes
- [x] `uv run pytest tests/test_collision_sphere_cache.py` passes
- [x] `uv run pytest tests/test_robots.py tests/test_solver_optional.py` passes
- [x] `uv run pytest` passes
- [x] `h1_2` and `teaarm` both instantiate `RobotCollision` from sphere decomposition at runtime
- [x] Over-approximation tests prove all sampled collision-surface points are inside at least one generated sphere (within configured epsilon)
- [x] Runtime emits explicit messages for approximation and cache behavior (miss/generate/hit/invalidate)

### Must Have
- Strict over-approximation path for mesh geometry.
- Deterministic decomposition output ordering and stable cache keying.
- Failure-safe fallback when decomposition backend is unavailable/fails.
- Safety fallback must guarantee link coverage when ballpark output fails conservative checks.
- Cache location and invalidation strategy are explicit, deterministic, and tested.
- Approximation step is explicitly visible to users via runtime messaging.
- No network required in tests (use local URDF strings/files and mocks).

### Must NOT Have (Guardrails)
- No migration work for `teleop_xr/ik/robots/franka.py` in this plan.
- No SPARROWS algorithm port.
- No manual verification steps.
- No expansion into unrelated IK objective tuning beyond collision representation swap.
- No committed runtime cache artifacts.

### Cache Position and Invalidation Strategy
- **Cache position**: `~/.cache/ram/collision_spheres/{robot_slug}/{cache_key}.json`
- **Metadata sidecar**: `~/.cache/ram/collision_spheres/{robot_slug}/{cache_key}.meta.json`
- **Cache key inputs**:
  - URDF content SHA256
  - collision mesh fingerprint digest (path + size + mtime + optional content hash)
  - ballpark version
  - pyroki version (or commit identifier if available)
  - generation parameters (spherize/refine/padding settings)
  - schema version string (e.g., `sphere_cache_v1`)
- **Invalidation triggers**:
  - URDF hash change
  - any link mesh fingerprint change
  - ballpark version change
  - pyroki version change
  - generation parameter change
  - schema version change
  - corrupted/partial cache file
- **User-visible runtime messages**:
  - cache miss: `Generating collision sphere approximation for {robot_slug}...`
  - cache hit: `Loaded cached collision sphere approximation for {robot_slug}`
  - cache invalidated: `Collision sphere cache invalidated ({reason}); regenerating...`

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan are verified by agent-executed commands/tests only.
> No acceptance criterion requires manual clicking, manual visual inspection, or user interaction.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: TDD
- **Framework**: `pytest`

### TDD Structure
For each implementation task:
1. **RED**: add/extend failing tests for the intended behavior.
2. **GREEN**: implement minimal code to pass those tests.
3. **REFACTOR**: clean up while keeping tests green.

### Agent-Executed QA Scenarios (applies to all tasks)
- Use `Bash` commands for test execution and evidence capture.
- Use deterministic fixtures/mocks to avoid flaky network/runtime dependencies.
- Capture artifacts in `.sisyphus/evidence/` as JSON/text logs.

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately):
- Task 1

Wave 2 (After Task 1):
- Task 2
- Task 3

Wave 3 (After Tasks 2 and 3):
- Task 4

Wave 4 (After Task 4):
- Task 5

Wave 5 (After Task 5):
- Task 6

Critical Path: Task 1 -> Task 2 -> Task 4 -> Task 5 -> Task 6

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|----------------------|
| 1 | None | 2, 3, 4, 5, 6 | None |
| 2 | 1 | 4, 5, 6 | 3 |
| 3 | 1 | 4, 5, 6 | 2 |
| 4 | 2, 3 | 5, 6 | None |
| 5 | 4 | 6 | None |
| 6 | 5 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|--------------------|
| 1 | 1 | `delegate_task(category="quick", load_skills=["code-reviewer","simplify"], run_in_background=false)` |
| 2 | 2, 3 | Run two dispatches in parallel: `category="unspecified-high"` for Task 2, `category="quick"` for Task 3 |
| 3 | 4 | `delegate_task(category="unspecified-high", load_skills=["code-reviewer","simplify"], run_in_background=false)` |
| 4 | 5 | `delegate_task(category="quick", load_skills=["code-reviewer"], run_in_background=false)` |
| 5 | 6 | `delegate_task(category="quick", load_skills=["code-reviewer","git-master"], run_in_background=false)` |

---

## TODOs

- [x] 1. Create RED contract tests for sphere decomposition integration

  **What to do**:
  - Add `tests/test_collision_sphere_contract.py` with failing tests that define expected behavior:
    - Pyroki sphere API presence gate (`from_sphere_decomposition`).
    - Decomposition schema shape and validation expectations.
    - Deterministic cache key generation for same URDF + params.
  - Add at least one failing test that currently fails before implementation.

  **Must NOT do**:
  - Do not call remote repos or network-dependent robot loaders.
  - Do not modify `franka` tests for this task.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: test-first scaffolding and assertions are localized edits.
  - **Skills**: `code-reviewer`, `simplify`
    - `code-reviewer`: keep assertions precise and maintainable.
    - `simplify`: keep fixtures minimal and deterministic.
  - **Skills Evaluated but Omitted**:
    - `git-master`: no commit orchestration needed at this step.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 1)
  - **Blocks**: 2, 3, 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `teleop_xr/ik/robots/h1_2.py:53` - current `RobotCollision.from_urdf(...)` baseline to replace.
  - `teleop_xr/ik/robots/teaarm.py:48` - same baseline in second target robot.
  - `tests/test_robots.py:81` - robot fixture/test style for URDF-string-based testing.
  - `tests/test_solver_optional.py:6` - solver invocation style to preserve.
  - `tests/test_ram.py:10` - deterministic cache fixture patterns (`tmp_path`, local files).
  - `uv.lock:3988` - confirms Pyroki source now points to `chungmin99/pyroki`.
  - `https://github.com/chungmin99/pyroki/pull/78` - expected API behavior and migration target.

  **Acceptance Criteria**:
  - [x] `tests/test_collision_sphere_contract.py` exists with RED tests.
  - [x] `uv run pytest tests/test_collision_sphere_contract.py` fails in RED phase due to missing implementation.
  - [x] Failure messages explicitly reference missing decomposition implementation or API wiring gap.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: RED tests fail for intended reason
    Tool: Bash
    Preconditions: New RED tests added, implementation not added yet
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_contract.py -q
      2. Assert: process exit code is non-zero
      3. Assert: failure output contains "from_sphere_decomposition" or "sphere decomposition"
      4. Save output: .sisyphus/evidence/task-1-red-failure.txt
    Expected Result: Controlled RED failure proves tests are meaningful
    Failure Indicators: Test unexpectedly passes or fails for unrelated import/network reasons
    Evidence: .sisyphus/evidence/task-1-red-failure.txt

  Scenario: Negative guard rejects malformed schema
    Tool: Bash
    Preconditions: RED tests include malformed dict case
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_contract.py::test_schema_rejects_malformed_input -q
      2. Assert: test fails in RED phase awaiting validator implementation
      3. Save output: .sisyphus/evidence/task-1-red-schema.txt
    Expected Result: Clear failing expectation for malformed input path
    Failure Indicators: Failure is unrelated to schema path
    Evidence: .sisyphus/evidence/task-1-red-schema.txt
  ```

  **Commit**: YES (groups with Task 2)
  - Message: `test(ik): add red tests for sphere decomposition contract`
  - Files: `tests/test_collision_sphere_contract.py`
  - Pre-commit: `uv run pytest tests/test_collision_sphere_contract.py -q`

---

- [x] 2. Implement ballpark-based decomposition engine with strict safety enforcement

  **What to do**:
  - Create `teleop_xr/ik/collision_spheres.py`.
  - Implement geometry extraction from URDF collision elements (mesh and primitives) in link-local frame.
  - For mesh collisions:
    - Load with `trimesh`.
    - Generate initial spheres via ballpark pipeline.
    - Run conservative post-check; if coverage fails, add/replace with link-level bounding sphere fallback.
  - For primitive collisions:
    - Use analytic enclosing spheres for box/cylinder/sphere.
  - Apply deterministic sorting and positive epsilon inflation for conservative safety margin.
  - Return exact Pyroki schema expected by `RobotCollision.from_sphere_decomposition(...)`.
  - Add ballpark dependency in `pyproject.toml`.

  **Must NOT do**:
  - Do not port SPARROWS algorithms.
  - Do not read visual meshes unless collision meshes are absent and fallback path explicitly requires it.
  - Do not change IK objective weights beyond collision construction.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: geometric algorithm implementation with transform correctness and fallback paths.
  - **Skills**: `code-reviewer`, `simplify`
    - `code-reviewer`: enforce robust edge-case handling.
    - `simplify`: avoid over-engineering and keep API narrow.
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: no UI work involved.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: 4, 5, 6
  - **Blocked By**: 1

  **References**:
  - `pyproject.toml:36` - current dependency list location for ballpark dependency addition.
  - `teleop_xr/ram.py:86` - existing path/URI normalization context (important for mesh path resolution assumptions).
  - `teleop_xr/ram.py:163` - resource loading contract currently used by robot constructors.
  - `teleop_xr/ik/robot.py:116` - `build_costs` interface that downstream integration must preserve.
  - `https://github.com/chungmin99/pyroki/pull/78` - sphere collision API expectations.
  - `https://github.com/chungmin99/ballpark` - ballpark generation approach and parameterization.
  - `https://github.com/mikedh/trimesh` - mesh loading and bounding sphere utilities.

  **Acceptance Criteria**:
  - [x] `teleop_xr/ik/collision_spheres.py` created with public function for runtime decomposition generation.
  - [x] RED tests from Task 1 pass GREEN for schema and deterministic key assertions.
  - [x] `uv run pytest tests/test_collision_sphere_contract.py -q` passes.
  - [x] At least one test verifies that sampled link-surface points are covered by generated spheres.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: Mesh decomposition produces valid Pyroki schema
    Tool: Bash
    Preconditions: collision_spheres module implemented, synthetic mesh fixture available
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_generation.py::test_mesh_decomposition_schema_valid -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-2-schema-pass.txt
    Expected Result: Generated dict has centers/radii with matching lengths and non-negative radii
    Failure Indicators: Key mismatch, empty geometry, or invalid radii
    Evidence: .sisyphus/evidence/task-2-schema-pass.txt

  Scenario: Ballpark generation failure triggers deterministic fallback
    Tool: Bash
    Preconditions: test monkeypatches ballpark call to fail
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_generation.py::test_ballpark_failure_falls_back_to_single_enclosing_sphere -q
      2. Assert: exit code 0
      3. Assert: fallback path assertion passes and output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-2-fallback-pass.txt
    Expected Result: Fallback sphere is produced and schema remains valid
    Failure Indicators: Exception escapes fallback path or no spheres returned
    Evidence: .sisyphus/evidence/task-2-fallback-pass.txt
  ```

  **Commit**: YES (with Task 1)
  - Message: `feat(ik): add ballpark-based sphere decomposition engine`
  - Files: `teleop_xr/ik/collision_spheres.py`, `pyproject.toml`, `tests/test_collision_sphere_contract.py`, `tests/test_collision_sphere_generation.py`
  - Pre-commit: `uv run pytest tests/test_collision_sphere_contract.py tests/test_collision_sphere_generation.py -q`

---

- [x] 3. Add deterministic decomposition cache with lock-safe persistence

  **What to do**:
  - Create `teleop_xr/ik/collision_sphere_cache.py`.
  - Implement cache path under RAM root: `~/.cache/ram/collision_spheres/{robot_slug}/`.
  - Store both payload and metadata:
    - `{cache_key}.json` for centers/radii
    - `{cache_key}.meta.json` for fingerprint/version/parameter metadata
  - Cache key must include:
    - URDF content hash
    - collision mesh fingerprint digest
    - decomposition algorithm version string
    - ballpark parameter set
    - pyroki version/commit identifier
  - Implement explicit invalidation reason mapping (e.g., `urdf_changed`, `mesh_changed`, `params_changed`, `version_changed`, `corrupt_cache`).
  - Use `filelock` for concurrent safety and atomic write strategy.
  - Validate cached JSON before reuse; recompute if invalid/corrupt.
  - Expose explicit status message strings for `cache miss`, `cache hit`, and `cache invalidated`.

  **Must NOT do**:
  - Do not commit generated cache files.
  - Do not silently ignore corrupted cache entries without recompute.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: focused utility module with deterministic IO behavior.
  - **Skills**: `code-reviewer`, `simplify`
    - `code-reviewer`: lock/atomicity correctness.
    - `simplify`: avoid complex cache abstractions.
  - **Skills Evaluated but Omitted**:
    - `git-master`: not required during active implementation.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 2)
  - **Blocks**: 4, 5, 6
  - **Blocked By**: 1

  **References**:
  - `teleop_xr/ram.py:71` - canonical cache root provider.
  - `teleop_xr/ram.py:119` - lock usage pattern with `FileLock`.
  - `tests/test_ram.py:10` - tmp cache fixture and local-path testing style.
  - `tests/test_ram.py:114` - existing cache root assertions.

  **Acceptance Criteria**:
  - [x] `tests/test_collision_sphere_cache.py` added and failing first (RED).
  - [x] Cache hit path verified to skip decomposition call on second invocation.
  - [x] Corrupted cache file path verified to recompute and heal cache.
  - [x] Invalidation reason is emitted and asserted in tests when cache becomes stale.
  - [x] Cache hit/miss/invalidation messages are asserted in tests.
  - [x] `uv run pytest tests/test_collision_sphere_cache.py -q` passes.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: Second load is cache hit
    Tool: Bash
    Preconditions: cache module implemented, decomposition call is spy-able in tests
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_cache.py::test_second_call_uses_cache_hit -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-3-cache-hit.txt
    Expected Result: second call does not invoke sphere generation backend
    Failure Indicators: backend call count increments on second call
    Evidence: .sisyphus/evidence/task-3-cache-hit.txt

  Scenario: Corrupted cache triggers recompute
    Tool: Bash
    Preconditions: test writes invalid JSON to expected cache file
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_cache.py::test_corrupt_cache_recomputed -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-3-cache-repair.txt
    Expected Result: invalid cache is not trusted and valid cache is rebuilt
    Failure Indicators: crash on JSON parse or stale invalid cache reused
    Evidence: .sisyphus/evidence/task-3-cache-repair.txt

  Scenario: Stale cache prints explicit invalidation reason
    Tool: Bash
    Preconditions: test mutates metadata input (e.g., params hash) between two loads
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_cache.py::test_stale_cache_emits_invalidation_reason -q
      2. Assert: exit code 0
      3. Assert: output contains "cache invalidated" and reason token
      4. Save output: .sisyphus/evidence/task-3-invalidation-message.txt
    Expected Result: user-visible invalidation reason appears and recompute occurs
    Failure Indicators: stale cache reused silently or reason missing
    Evidence: .sisyphus/evidence/task-3-invalidation-message.txt
  ```

  **Commit**: YES
  - Message: `feat(ik): add lock-safe cache for sphere decompositions`
  - Files: `teleop_xr/ik/collision_sphere_cache.py`, `tests/test_collision_sphere_cache.py`
  - Pre-commit: `uv run pytest tests/test_collision_sphere_cache.py -q`

---

- [x] 4. Integrate sphere decomposition into `h1_2` and `teaarm` collision construction

  **What to do**:
  - Update `teleop_xr/ik/robots/h1_2.py` and `teleop_xr/ik/robots/teaarm.py`:
    - Replace `RobotCollision.from_urdf(urdf)` path with runtime decomposition + `RobotCollision.from_sphere_decomposition(...)`.
    - Wire cache-backed generation into initialization.
    - Surface approximation lifecycle messages to user-facing logs (`cache miss`, `generating`, `cache hit`, `invalidated`).
    - Re-enable self-collision cost blocks in `build_costs(...)` using current per-robot margins/weights unless tests prove adjustment required.
  - Keep existing non-collision cost terms unchanged.

  **Must NOT do**:
  - Do not modify `teleop_xr/ik/robots/franka.py`.
  - Do not change solver algorithm or controller behavior.
  - Do not alter unrelated IK objective scales.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: touches runtime robot construction and optimization cost composition.
  - **Skills**: `code-reviewer`, `simplify`
    - `code-reviewer`: preserve behavior while changing collision backend.
    - `simplify`: avoid broad refactors in critical robot classes.
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: irrelevant to backend IK path.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: 5, 6
  - **Blocked By**: 2, 3

  **References**:
  - `teleop_xr/ik/robots/h1_2.py:52` - current robot model creation baseline.
  - `teleop_xr/ik/robots/h1_2.py:53` - current collision model creation to replace.
  - `teleop_xr/ik/robots/h1_2.py:142` - commented self-collision cost block to re-enable.
  - `teleop_xr/ik/robots/teaarm.py:47` - current robot model creation baseline.
  - `teleop_xr/ik/robots/teaarm.py:48` - current collision model creation to replace.
  - `teleop_xr/ik/robots/teaarm.py:198` - commented self-collision cost block to re-enable.
  - `teleop_xr/ik/solver.py:58` - solver call path that consumes robot-built costs.

  **Acceptance Criteria**:
  - [x] Both robot classes initialize collision via sphere decomposition API.
  - [x] Both robot classes include active self-collision cost entries in `build_costs`.
  - [x] Robot initialization path emits explicit approximation/cache status messages.
  - [x] `uv run pytest tests/test_robots.py -q` passes.
  - [x] `uv run pytest tests/test_solver_optional.py -q` passes.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: TeaArm robot builds costs with sphere collision path active
    Tool: Bash
    Preconditions: robot integration complete, tests patch remote paths as needed
    Steps:
      1. Run: uv run pytest tests/test_robots.py::test_teaarm_robot -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-4-teaarm-pass.txt
    Expected Result: TeaArm initialization and build_costs succeed with new collision pipeline
    Failure Indicators: failure in collision model initialization or cost construction
    Evidence: .sisyphus/evidence/task-4-teaarm-pass.txt

  Scenario: Negative path with unsupported collision geometry handled gracefully
    Tool: Bash
    Preconditions: dedicated test injects unsupported geometry node
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_generation.py::test_unsupported_collision_geometry_degrades_safely -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-4-unsupported-geom.txt
    Expected Result: unsupported geometry does not crash initialization and fallback path remains valid
    Failure Indicators: unhandled exception or empty decomposition for all links
    Evidence: .sisyphus/evidence/task-4-unsupported-geom.txt

  Scenario: User-visible message sequence on cache miss then hit
    Tool: Bash
    Preconditions: first run uses empty cache, second run reuses generated cache
    Steps:
      1. Run: uv run pytest tests/test_robots.py::test_collision_sphere_messages_miss_then_hit -q
      2. Assert: exit code 0
      3. Assert: captured logs contain "Generating collision sphere approximation" then "Loaded cached collision sphere approximation"
      4. Save output: .sisyphus/evidence/task-4-message-sequence.txt
    Expected Result: approximation step is explicit to user and cache benefit is visible
    Failure Indicators: messages absent, ambiguous, or only debug-level hidden output
    Evidence: .sisyphus/evidence/task-4-message-sequence.txt
  ```

  **Commit**: YES
  - Message: `feat(ik): switch h1 and teaarm to dynamic sphere collisions`
  - Files: `teleop_xr/ik/robots/h1_2.py`, `teleop_xr/ik/robots/teaarm.py`
  - Pre-commit: `uv run pytest tests/test_robots.py tests/test_solver_optional.py -q`

---

- [x] 5. Complete GREEN coverage for over-approximation, determinism, and regression safety

  **What to do**:
  - Finalize/extend tests to cover:
    - surface-point containment (strict over-approximation)
    - deterministic ordering and cache key stability
    - multiple collision elements per link
    - primitive-only links
    - decomposition timeout/failure fallback
    - no regression in solver optional-target behavior
  - Ensure all tests run without network by mocking RAM loaders where necessary.

  **Must NOT do**:
  - Do not weaken assertions to avoid legitimate geometry issues.
  - Do not remove existing solver tests to force pass.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: test consolidation and assertion hardening.
  - **Skills**: `code-reviewer`, `simplify`
    - `code-reviewer`: tighten edge-case assertions.
    - `simplify`: avoid brittle fixture complexity.
  - **Skills Evaluated but Omitted**:
    - `git-master`: not needed while tests are still evolving.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential)
  - **Blocks**: 6
  - **Blocked By**: 4

  **References**:
  - `tests/test_robots.py:126` - H1 robot test pattern and mocking approach.
  - `tests/test_solver_optional.py:6` - optional target solve behavior baseline.
  - `tests/test_ik_api.py:45` - abstract interface expectations for `build_costs`.
  - `tests/test_ram.py:259` - no-package-resolution behavior pattern for local path correctness.

  **Acceptance Criteria**:
  - [x] `uv run pytest tests/test_collision_sphere_generation.py -q` passes.
  - [x] `uv run pytest tests/test_collision_sphere_cache.py -q` passes.
  - [x] `uv run pytest tests/test_robots.py tests/test_solver_optional.py -q` passes.
  - [x] Over-approximation tests include both happy path and negative/error path.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: Over-approximation containment passes on synthetic mesh
    Tool: Bash
    Preconditions: generation tests implemented with deterministic fixture mesh
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_generation.py::test_all_sampled_surface_points_are_contained -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-5-containment-pass.txt
    Expected Result: every sampled point satisfies containment inequality within epsilon
    Failure Indicators: uncovered sampled points or numeric instability
    Evidence: .sisyphus/evidence/task-5-containment-pass.txt

  Scenario: Legacy API mismatch fails fast with clear message
    Tool: Bash
    Preconditions: test monkeypatch removes `from_sphere_decomposition` attribute
    Steps:
      1. Run: uv run pytest tests/test_collision_sphere_contract.py::test_missing_pyroki_sphere_api_fails_fast -q
      2. Assert: exit code 0
      3. Assert: output contains "1 passed"
      4. Save output: .sisyphus/evidence/task-5-api-guard-pass.txt
    Expected Result: code raises explicit runtime/config error rather than obscure failure later
    Failure Indicators: silent fallback to wrong API or vague exception
    Evidence: .sisyphus/evidence/task-5-api-guard-pass.txt
  ```

  **Commit**: YES
  - Message: `test(ik): cover overapprox, cache, and regression scenarios`
  - Files: `tests/test_collision_sphere_generation.py`, `tests/test_collision_sphere_cache.py`, `tests/test_robots.py`, `tests/test_solver_optional.py`
  - Pre-commit: `uv run pytest tests/test_collision_sphere_generation.py tests/test_collision_sphere_cache.py tests/test_robots.py tests/test_solver_optional.py -q`

---

- [x] 6. Refactor, parameter pinning, and full-suite verification

  **What to do**:
  - Refactor decomposition/caching code for clarity without behavior changes.
  - Pin and centralize decomposition parameters (resolution, max hulls, epsilon inflation, timeout behavior) in one module-level config structure.
  - Ensure logs/errors are actionable for fallback and cache paths.
  - Run full suite and finalize evidence bundle.

  **Must NOT do**:
  - Do not introduce new robot support.
  - Do not change collision business logic after tests are green unless required by failing evidence.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: final stabilization and regression verification.
  - **Skills**: `code-reviewer`, `git-master`
    - `code-reviewer`: final correctness pass.
    - `git-master`: clean commit slicing and message quality.
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: no frontend scope.

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (final sequential)
  - **Blocks**: None
  - **Blocked By**: 5

  **References**:
  - `pyproject.toml:82` - pytest config and coverage defaults.
  - `README.md` - development test command conventions.
  - `.github/copilot-instructions.md` - repository-standard test command guidance.

  **Acceptance Criteria**:
  - [x] `uv run pytest` passes.
  - [x] Evidence files exist for each prior task scenario under `.sisyphus/evidence/`.
  - [x] No `franka` code changes included.
  - [x] Collision generation and cache modules have deterministic parameter configuration in a single location.

  **Agent-Executed QA Scenarios**:
  ```text
  Scenario: Full regression suite passes
    Tool: Bash
    Preconditions: All prior tasks completed and committed locally
    Steps:
      1. Run: uv run pytest
      2. Assert: exit code 0
      3. Assert: output contains "passed" and no failed tests
      4. Save output: .sisyphus/evidence/task-6-full-pytest.txt
    Expected Result: No regressions in backend test suite
    Failure Indicators: any test failure, timeout, or skipped critical decomposition tests
    Evidence: .sisyphus/evidence/task-6-full-pytest.txt

  Scenario: Negative verification ensures franka untouched
    Tool: Bash
    Preconditions: local changes present
    Steps:
      1. Run: git diff --name-only
      2. Assert: output does NOT include teleop_xr/ik/robots/franka.py
      3. Save output: .sisyphus/evidence/task-6-scope-guard.txt
    Expected Result: scope guardrail preserved
    Failure Indicators: franka file modified
    Evidence: .sisyphus/evidence/task-6-scope-guard.txt
  ```

  **Commit**: YES
  - Message: `chore(ik): finalize sphere collision migration verification`
  - Files: all changed migration files from tasks 1-6
  - Pre-commit: `uv run pytest`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 2 | `feat(ik): add ballpark-based sphere decomposition engine` | `teleop_xr/ik/collision_spheres.py`, `pyproject.toml`, contract/generation tests | targeted pytest for contract/generation |
| 3 | `feat(ik): add lock-safe cache for sphere decompositions` | `teleop_xr/ik/collision_sphere_cache.py`, cache tests | targeted pytest for cache |
| 4 | `feat(ik): switch h1 and teaarm to dynamic sphere collisions` | `teleop_xr/ik/robots/h1_2.py`, `teleop_xr/ik/robots/teaarm.py` | robot + solver optional tests |
| 5 | `test(ik): cover overapprox, cache, and regression scenarios` | updated/new tests | targeted pytest bundle |
| 6 | `chore(ik): finalize sphere collision migration verification` | all remaining migration deltas | full `uv run pytest` |

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/test_collision_sphere_contract.py
uv run pytest tests/test_collision_sphere_generation.py
uv run pytest tests/test_collision_sphere_cache.py
uv run pytest tests/test_robots.py tests/test_solver_optional.py
uv run pytest
```

### Final Checklist
- [x] Sphere decomposition generated dynamically at runtime for `h1_2` and `teaarm`.
- [x] Generated spheres are verified over-approximations of collision geometry.
- [x] Cache is deterministic, lock-safe, and validated on read.
- [x] Self-collision cost is active for both migrated robots.
- [x] No `franka` modifications in this migration.
- [x] All automated tests pass with no manual verification steps.
