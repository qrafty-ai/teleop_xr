# Plan: Robot Asset Manager (RAM) Module

## TL;DR

> **Quick Summary**: Create a `ram` (Robot Asset Manager) helper module in `teleop_xr` to fetch robot assets from git, cache them locally, and process URDF/Xacro files with `package://` resolution.
>
> **Deliverables**:
> - `teleop_xr/ram.py`: Core module.
> - `tests/test_ram.py`: Comprehensive tests with local git fixtures.
> - `pyproject.toml`: Add `gitpython` and `xacro` dependencies.
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Dependencies -> `ram.py` implementation -> Tests

---

## Context

### Original Request
User wants a `ram` helper module to:
1. Fetch a specific folder from a git repo into a dedicated cache.
2. Return the URDF file path.
3. Support Xacro auto-generation with user args.
4. Handle `package://` prefixes by replacing them with relative paths (for ROS-based repos).

### Interview Summary
**Key Decisions**:
- **Location**: `teleop_xr/ram.py`
- **Cache**: `~/.cache/ram` (user home directory, XDG compliant)
- **Dependencies**: Add `gitpython` and `xacro` to `pyproject.toml`
- **Xacro Strategy**: Use `xacro` python library (pip installed)
- **Testing**: TDD with `pytest`, using local git repos as fixtures.
- **URI Handling**: `package://` replaced with relative paths to cached files.

### Metis Review
**Identified Gaps** (addressed):
- **Concurrency**: Added atomic write/locking requirement for cache operations.
- **Xacro `$(find)`**: Added substitution handling for common ROS patterns (best effort).
- **Test Isolation**: Explicitly required using local temp git repos for tests (no network dependency for tests).

---

## Work Objectives

### Core Objective
Enable dynamic fetching and processing of robot descriptions (URDF/Xacro) from git repositories without committing them to the source tree.

### Concrete Deliverables
- `teleop_xr/ram.py` containing `get_asset(...)` and `process_xacro(...)` functions.
- Updates to `pyproject.toml`.
- New test file `tests/test_ram.py`.

### Definition of Done
- [x] `ram.get_asset()` successfully clones/fetches a remote repo.
- [x] `package://` paths in URDFs are resolved to correct relative paths on disk.
- [x] Xacro files are converted to URDF with provided arguments.
- [x] Tests pass offline using local fixtures.

### Must Have
- Caching mechanism to avoid re-downloading if commit hash/branch hasn't changed (or if TTL is valid).
- Handling of `package://` replacement.
- Support for `git` authentication (via system git credentials/ssh agent - `gitpython` uses system git).

### Must NOT Have (Guardrails)
- **Full ROS dependency**: Do NOT require a full ROS installation (no `rospack`).
- **Complex Dependency Resolution**: RAM will not recursively fetch dependencies of dependencies.

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> ALL tasks must be verified by agents using provided commands.

### Test Decision
- **Infrastructure exists**: YES (`pytest`)
- **Automated tests**: YES (TDD)
- **Framework**: `pytest`

### Agent-Executed QA Scenarios

#### Scenario: Fetch and Process Xacro
**Tool**: `bash` (pytest)
**Steps**:
1. Run `uv run pytest tests/test_ram.py -k "test_fetch_and_process_xacro"`
2. Assert exit code 0.
**Evidence**: `tests/test_ram.py` output.

#### Scenario: Offline Cache Retrieval
**Tool**: `bash` (pytest)
**Steps**:
1. Run `uv run pytest tests/test_ram.py -k "test_cache_retrieval"`
2. Assert exit code 0.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Add Dependencies (pyproject.toml)
└── Task 2: Create Test Fixtures (tests/test_ram.py skeleton)

Wave 2 (After Wave 1):
└── Task 3: Implement RAM Module (teleop_xr/ram.py)
```

---

## TODOs

- [x] 1. Add RAM Dependencies

  **What to do**:
  - Add `gitpython` and `xacro` to `dependencies` in `pyproject.toml`.
  - Run `uv sync` (or verify installation commands).

  **Must NOT do**:
  - Remove existing dependencies.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`python`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3

  **Acceptance Criteria**:
  - [x] `grep "gitpython" pyproject.toml` returns match
  - [x] `grep "xacro" pyproject.toml` returns match
  - [x] `uv run python -c "import git; import xacro; print('ok')"` returns "ok"

- [x] 2. Create RAM Test Suite (TDD)

  **What to do**:
  - Create `tests/test_ram.py`.
  - Implement fixtures for:
    - `temp_git_repo`: Creates a local git repo with sample URDF/Xacro/Meshes in a temp dir.
    - `mock_cache_dir`: Sets up a temp dir for RAM cache.
  - Implement test cases (initially failing or skipped):
    - `test_fetch_git_repo`: Verifies cloning.
    - `test_package_path_replacement`: Verifies `package://` -> relative path.
    - `test_xacro_conversion`: Verifies xacro -> urdf.

  **Recommended Agent Profile**:
  - **Category**: `python-developer`
  - **Skills**: [`python`, `git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3

  **Acceptance Criteria**:
  - [x] `tests/test_ram.py` exists.
  - [x] `uv run pytest tests/test_ram.py` runs (may fail or skip, but files exist).

- [x] 3. Implement RAM Module Core

  **What to do**:
  - Create `teleop_xr/ram.py`.
  - Implement `get_repo(repo_url, branch, cache_dir)` using `gitpython`.
    - Should clone if not exists, pull if exists.
  - Implement `process_xacro(xacro_path, mappings, output_path)`.
    - Handle `package://` replacement in content BEFORE or AFTER xacro processing (usually before if xacro includes use package://, or rely on xacro's substitution args).
    - **Strategy**:
        1. If Xacro file, run `xacro` processing.
        2. In the resulting URDF string, regex replace `package://<pkg_name>/` with relative path to the file in cache.
  - Implement `get_resource(repo_url, path_inside_repo, ...)` main entry point.

  **References**:
  - `src/onshape2xacro/visualize_export.py` (existing URDF logic in repo).
  - `teleop_xr/robot_vis.py` (existing asset serving logic).

  **Recommended Agent Profile**:
  - **Category**: `python-developer`
  - **Skills**: [`python`, `git-master`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocked By**: Task 1, Task 2

  **Acceptance Criteria**:
  - [x] `uv run pytest tests/test_ram.py` passes.
  - [x] Function `get_resource` returns valid Path object.
  - [x] `package://` paths are correctly resolved in output.

---

## Success Criteria

### Verification Commands
```bash
uv run pytest tests/test_ram.py
```

### Final Checklist
- [x] `gitpython` and `xacro` installed.
- [x] `teleop_xr/ram.py` implements caching and processing.
- [x] Tests cover xacro conversion and path replacement.
