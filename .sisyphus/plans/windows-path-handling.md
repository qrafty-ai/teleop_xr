# Systematic Windows Path Handling for ram.py & robot_vis.py

## TL;DR

> **Quick Summary**: Replace the ad-hoc `.replace('\\', '/')` approach from PR #55 with a systematic `.as_posix()` strategy applied at URDF-embedding boundaries only, clean up robot_vis.py path logic, add targeted unit tests using `PureWindowsPath`, and add Windows CI runner.
>
> **Deliverables**:
> - Fixed `ram.py` with `.as_posix()` at all URDF-embed sites
> - Cleaned `robot_vis.py` path stripping logic (cross-platform compatible)
> - New `tests/test_ram_windows_paths.py` with `PureWindowsPath`-based tests
> - Updated `.github/workflows/ci.yml` with `windows-latest` runner
> - PR #55's openarm_description hack NOT included
>
> **Estimated Effort**: Short (2-3 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 5

---

## Context

### Original Request
PR #55 (https://github.com/qrafty-ai/teleop_xr/pull/55) attempts to fix Windows path issues in ram.py by scattering `.replace('\\', '/')` calls. User wants a cleaner, systematic approach instead.

### Interview Summary
**Key Discussions**:
- PR #55's approach is ad-hoc: scattered `.replace('\\', '/')` in 4 places, misses sites, includes unrelated hack
- `ram.py` has two path usage categories: FILESYSTEM (native OS, correct) and URDF-EMBED (must be forward slashes, broken on Windows)
- `_resolve_package` returns flow to BOTH contexts — fix must happen at embedding boundary, not resolution point
- `robot_vis.py` has the same bug class: `os.path.abspath()` backslashes vs URDF forward-slash content
- User wants robot_vis.py cleaned: should only handle relative paths since package:// removal is ram.py's job
- Drop the hardcoded openarm_description hack from PR #55
- Add Windows CI runner
- **CI Fix**: Exclude `cuda` extra on Windows runners as `jax[cuda12]` lacks Windows wheels for the environment.

**Research Findings**:
- `Path.as_posix()` is the canonical Python method — no-op on Linux, converts backslashes on Windows
- `PureWindowsPath` can simulate Windows path behavior on Linux for testing
- Existing test suite: 530+ lines in test_ram.py, CI is Linux-only

### Metis Review
**Identified Gaps** (addressed):
- Metis noted `as_posix()` on absolute Windows paths produces `C:/...` with drive letters — this is fine for our use case since URDF consumers in this project are the WebXR frontend (three.js URDFLoader) and viser, both of which accept forward-slash absolute paths
- Metis flagged UNC paths (`\\server\share\...`) — out of scope, this is a local development tool not used on network shares
- Metis noted potential mixed-slash inputs in URDF source — the regex in `_replace_package_uris` matches `package://` URIs which always use forward slashes, so this is not a concern
- Metis suggested checking for non-`package://` schemes (file://, http://) — the existing regex only matches `package://`, other schemes are correctly left alone
- Metis flagged that `_resolve_package` should continue returning native `Path`-compatible strings — agreed, this is a core guardrail

---

## Work Objectives

### Core Objective
Ensure all paths embedded into URDF/XML content use forward slashes regardless of OS, by applying `Path.as_posix()` at embedding boundaries. Clean robot_vis.py path logic. Verify with cross-platform tests and Windows CI.

### Concrete Deliverables
- Modified `teleop_xr/ram.py` — 3 URDF-embed sites fixed
- Modified `teleop_xr/robot_vis.py` — path stripping logic made cross-platform
- New `tests/test_ram_windows_paths.py` — PureWindowsPath-based path tests
- Modified `.github/workflows/ci.yml` — windows-latest added to test matrix

### Definition of Done
- [x] `python -m pytest -q` passes on Linux
- [x] No backslashes (`\`) appear in any URDF output strings produced by ram.py
- [x] robot_vis.py path stripping works when URDF content has forward-slash paths and OS uses backslashes
- [x] CI runs tests on both ubuntu-latest and windows-latest

### Must Have
- `.as_posix()` at URDF-embed boundaries (ram.py lines 116, 119, 155)
- Cross-platform path comparison in robot_vis.py
- Unit tests that verify no backslashes in URDF output using PureWindowsPath simulation
- Windows CI runner

### Must NOT Have (Guardrails)
- NO scattered `.replace('\\', '/')` calls — use `.as_posix()` at embed points only
- NO changes to `_resolve_package` return values — it must continue returning native-path strings for xacro/filesystem use
- NO early posix-ification of filesystem paths (don't convert paths before they reach the embedding boundary)
- NO hardcoded package name hacks (drop the openarm_description special case from PR #55)
- NO scope expansion to other files (teleop_xr/__init__.py, ik/robots/*.py use os.path for filesystem only)
- NO changes to path semantics (absolute vs relative, package root resolution logic)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: Tests-after (targeted path tests)
- **Framework**: pytest (already configured in pyproject.toml)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Python module** | Bash (pytest) | Run tests, assert pass |
| **URDF output** | Bash (pytest assertions) | Assert no backslashes in output |
| **CI config** | Bash (yaml validation) | Validate yaml syntax |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Fix ram.py URDF-embed paths with .as_posix()
├── Task 2: Clean robot_vis.py path logic
└── Task 4: Add Windows CI runner to ci.yml

Wave 2 (After Wave 1):
├── Task 3: Add Windows path unit tests
└── Task 5: Final integration verification

Critical Path: Task 1 → Task 3 → Task 5
Parallel Speedup: ~30% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 5 | 2, 4 |
| 2 | None | 3, 5 | 1, 4 |
| 3 | 1, 2 | 5 | 4 |
| 4 | None | 5 | 1, 2 |
| 5 | 1, 2, 3, 4 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2, 4 | task(category="quick", load_skills=[], ...) for each |
| 2 | 3, 5 | task(category="unspecified-low", load_skills=[], ...) |

---

## TODOs

- [x] 1. Fix ram.py URDF-embed paths with `.as_posix()`
- [x] 2. Clean robot_vis.py path logic
- [x] 3. Add Windows path unit tests
- [x] 4. Add Windows CI runner to ci.yml
- [x] 5. Final integration verification

  **What to do**:
  - Run the full test suite to verify no regressions
  - Verify the complete set of changes works together
  - Review that no `.replace('\\', '/')` calls remain in modified files
  - Verify the commit history is clean

  **Must NOT do**:
  - Do not push to remote
  - Do not merge PR #55

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification only — no code changes
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (final, after all other tasks)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1, 2, 3, 4

  **References**:

  **All modified files**:
  - `teleop_xr/ram.py` — URDF-embed path fixes
  - `teleop_xr/robot_vis.py` — path stripping cleanup
  - `tests/test_ram_windows_paths.py` — new test file
  - `.github/workflows/ci.yml` — Windows CI

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full test suite passes
    Tool: Bash (pytest)
    Preconditions: All tasks 1-4 completed
    Steps:
      1. Run: python -m pytest tests/ -x --tb=short
      2. Assert: Exit code 0
      3. Assert: No test failures or errors
    Expected Result: All tests pass
    Evidence: Full pytest output captured to .sisyphus/evidence/task-5-full-suite.txt

  Scenario: No .replace backslash calls remain in modified files
    Tool: Bash (grep)
    Preconditions: All changes applied
    Steps:
      1. Run: grep -rn "replace.*\\\\\\\\.*/" teleop_xr/ram.py teleop_xr/robot_vis.py
      2. Assert: No matches found (exit code 1)
      3. Run: grep -c "as_posix" teleop_xr/ram.py
      4. Assert: Count >= 3 (at least 3 embed sites converted)
    Expected Result: Clean codebase with .as_posix() instead of .replace
    Evidence: grep output captured

  Scenario: Verify URDF output format
    Tool: Bash (python one-liner)
    Preconditions: All changes applied
    Steps:
      1. Run: python -c "
         from pathlib import PureWindowsPath
         p = PureWindowsPath(r'C:\Users\dev\repo\mesh.stl')
         assert p.as_posix() == 'C:/Users/dev/repo/mesh.stl'
         print('as_posix() works correctly')
         "
      2. Assert: prints "as_posix() works correctly"
    Expected Result: Confirms the core mechanism works
    Evidence: Output captured
  ```

  **Commit**: NO (verification only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1+2 | `fix(ram,robot_vis): use .as_posix() for URDF-embedded paths on Windows` | `teleop_xr/ram.py`, `teleop_xr/robot_vis.py` | `python -m pytest tests/test_ram.py tests/test_ram_local.py -x` |
| 3 | `test(ram): add Windows path simulation tests using PureWindowsPath` | `tests/test_ram_windows_paths.py` | `python -m pytest tests/test_ram_windows_paths.py -x` |
| 4 | `ci: add Windows runner to test matrix` | `.github/workflows/ci.yml` | YAML validation |

---

## Success Criteria

### Verification Commands
```bash
python -m pytest tests/ -x --tb=short  # Expected: all tests pass
grep -rn "replace.*\\\\.*/" teleop_xr/ram.py  # Expected: no matches
grep -c "as_posix" teleop_xr/ram.py  # Expected: >= 3
python -c "from pathlib import PureWindowsPath; p = PureWindowsPath(r'C:\a\b'); assert '\\\\' not in p.as_posix()"  # Expected: passes
```

### Final Checklist
- [x] All "Must Have" present (`.as_posix()` at 3 ram.py sites, robot_vis.py fix, tests, Windows CI)
- [x] All "Must NOT Have" absent (no `.replace('\\', '/')`, no openarm hack, no early posix-ification)
- [x] All tests pass on Linux
- [x] CI config includes windows-latest for tests job
