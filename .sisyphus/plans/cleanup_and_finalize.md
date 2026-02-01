# Cleanup and Finalize WebXR Fixes

## TL;DR

> **Quick Summary**: Remove debug artifacts (green box) from the verification test, verify the correct rendering (Dashboard + Blue Robot Placeholder), and commit the source code fixes.
>
> **Deliverables**:
> - Cleaned up `tests/final_verification.spec.ts`
> - `final_verification.png` showing Dashboard + Blue Robot
> - Git commit with WebXR fixes
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential
> **Critical Path**: Cleanup Test → Run Verification → Commit

---

## Context

### Original Request
The user wants to finalize the "See Dashboard & Model" task. We fixed the rendering (confirmed by green box), but now need to remove the debug green box, verify the real components (dashboard and fallback robot) are visible, and commit the fixes.

### Interview Summary
**Key Discussions**:
- **Artifacts**: The green box in `tests/final_verification.spec.ts` must go.
- **Robot Model**: The blue placeholder box in `webxr/src/components/robot-model.ts` is the intended fallback visual. We keep it.
- **Commit Scope**: We have pending changes in `webxr/` (index.ts, robot-model.ts, etc.) that represent the "fixes". These will be committed. The test file is untracked and will remain local.

### Metis Review
**Identified Gaps** (addressed):
- **Commit Scope**: Verified via `git status` that `webxr/` has the relevant modifications.
- **Verification**: We must verify the screenshot *after* cleanup to ensure the dashboard/robot are actually visible without the green box.

---

## Work Objectives

### Core Objective
Clean up debug code and commit the functional fixes for WebXR component registration.

### Concrete Deliverables
- [ ] Modified `tests/final_verification.spec.ts` (green box removed)
- [ ] New `final_verification.png` (verified visual)
- [ ] Git commit `fix(webxr): resolve component registration and double-loading issues`

### Definition of Done
- [ ] `npx playwright test tests/final_verification.spec.ts` passes
- [ ] Screenshot shows Dark Dashboard + Blue Box
- [ ] `git status` shows clean `webxr/` directory (committed)

### Must Have
- Blue box visible (Robot placeholder)
- Dashboard visible (Dark plane with text)

### Must NOT Have (Guardrails)
- Do NOT remove the blue box from `robot-model.ts`.
- Do NOT commit the untracked test files.

---

## Verification Strategy

### Automated Verification Only (NO User Intervention)

**By Deliverable Type:**

| Type | Verification Tool | Automated Procedure |
|------|------------------|---------------------|
| **Visual/Test** | Bash (Playwright) | Run the test, check exit code 0. Screenshot is generated for evidence. |
| **Commit** | Bash (Git) | Verify commit exists and working tree is clean (for tracked files). |

**Evidence Requirements:**
- Terminal output of test run.
- `final_verification.png` in `.sisyphus/evidence/`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Sequential):
├── Task 1: Remove Green Box from Test (Edit)
├── Task 2: Run Final Verification (Test)
└── Task 3: Commit Fixes (Git)
```

### Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Cleanup is independent |
| Task 2 | Task 1 | Verify *after* cleanup |
| Task 3 | Task 2 | Commit only after verification passes |

---

## Tasks

### Task 1: Cleanup Verification Test
**Description**: Remove the code block that injects the green debug box from `tests/final_verification.spec.ts`.
**Delegation Recommendation**:
- Category: `quick` - Simple file edit.
- Skills: [`typescript-programmer`] - Understands the code structure to remove.
**Skills Evaluation**:
- INCLUDED `typescript-programmer`: To correctly identify and remove the TS code block.
- OMITTED `git-master`: Not needed for file editing.
**Depends On**: None
**Acceptance Criteria**:
- [ ] Lines 39-45 (green box injection) removed from `tests/final_verification.spec.ts`.

### Task 2: Run Final Verification
**Description**: Run the modified test to generate the final screenshot and verify the scene.
**Delegation Recommendation**:
- Category: `quick` - Running a command.
- Skills: [`agent-browser`, `playwright`] - Although running via bash, understanding playwright output is helpful.
**Skills Evaluation**:
- INCLUDED `playwright`: To parse test output if needed.
- OMITTED `visual-engineering`: We are trusting the test pass/fail.
**Depends On**: Task 1
**Acceptance Criteria**:
- [ ] Command `npx playwright test tests/final_verification.spec.ts` returns exit code 0.
- [ ] Screenshot `final_verification.png` is created.

### Task 3: Commit Fixes
**Description**: Commit the changes in `webxr/` that fixed the component registration issues.
**Delegation Recommendation**:
- Category: `quick` - Git operation.
- Skills: [`git-master`] - Essential for correct committing.
**Skills Evaluation**:
- INCLUDED `git-master`: For atomic commit and message formatting.
**Depends On**: Task 2
**Acceptance Criteria**:
- [ ] `git add webxr` executed.
- [ ] `git commit -m "fix(webxr): resolve component registration and double-loading issues"` executed.
- [ ] `git push` executed.

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 3 | `fix(webxr): resolve component registration and double-loading issues` | `webxr/` | Test passed in Task 2 |

---

## Success Criteria

### Verification Commands
```bash
npx playwright test tests/final_verification.spec.ts
git log -1 --oneline
```

### Final Checklist
- [ ] Green box gone from code
- [ ] Test passes
- [ ] Changes committed
