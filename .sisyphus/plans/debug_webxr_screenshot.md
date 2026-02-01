# Debug WebXR Screenshot & Rendering

## Context

**Goal**: Fix the blank (grey) screenshot issue in the WebXR dashboard/robot model view.
**Current Status**:
- Screenshot is capturing a grey background, implying the scene isn't rendering or the screenshot is taken too early.
- The user mentioned `<a-scene>` modifications, suggesting A-Frame usage, but the codebase structure (`webxr/src/`) hints at a Three.js/ECS stack.
- Playwright (likely running headless) often fails to initialize WebGL or renders blank canvases without specific flags.

**Approach**:
1. **Verify Framework**: Confirm if the app is A-Frame or pure Three.js to use the correct debug primitives.
2. **Diagnostic Instrumentation**: Add a visible DOM overlay (to prove the screenshot works) and a 3D primitive (to prove WebGL works).
3. **Robust Debug Script**: Create a Playwright script that handles headless WebGL, captures all console/network logs, and waits for a deterministic "ready" state.
4. **Fix & Verify**: Analyze logs to fix the root cause (headless config, asset 404s, or wait timing).

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Determines if we use A-Frame or Three.js code |
| Task 2 | Task 1 | Needs framework knowledge to inject correct 3D debug object |
| Task 3 | Task 2 | Requires the modified HTML to test rendering |
| Task 4 | Task 3 | Analyzing logs requires the script to have run |
| Task 5 | Task 4 | Fixes depend on the analysis |

## Parallel Execution Graph

**Wave 1 (Start immediately)**:
├── Task 1: Verify framework & Add DOM overlay (Independent)
└── Task 3: Create Playwright debug script (Independent of HTML changes)

**Wave 2 (After Wave 1)**:
└── Task 2: Add 3D debug primitive (Depends on Framework knowledge)

**Wave 3 (After Wave 2)**:
└── Task 4: Run diagnostics & Analyze logs

**Wave 4 (After Wave 3)**:
└── Task 5: Apply Fixes & Final Verification

## Tasks

### Task 1: Verify Framework & Add DOM Overlay
**Description**:
1. Read `webxr/index.html` to confirm if `<a-scene>` is present.
2. Add a **DOM-based debug overlay** (red HTML div, fixed position, z-index 9999) to `webxr/index.html`.
   - This proves the screenshot tool works even if WebGL fails.
   - Text content: "DEBUG OVERLAY".
**Delegation Recommendation**:
- Category: `visual-engineering` - HTML/CSS inspection and modification.
- Skills: [`frontend-ui-ux`, `grep`] - To identify framework and edit HTML.
**Skills Evaluation**:
- ✅ `frontend-ui-ux`: For HTML injection.
- ❌ `playwright`: Not running browser yet.
**Acceptance Criteria**:
- Framework identified (A-Frame vs Three.js).
- `index.html` contains `<div id="debug-overlay" ...>`.

### Task 2: Add 3D Debug Primitive
**Description**:
Based on Task 1's finding, add a 3D primitive to the scene to test WebGL rendering.
- **If A-Frame**: Add `<a-box color="blue" position="0 1.5 -2" material="shader: flat"></a-box>`.
- **If Three.js**: Inject a simple `Mesh` (BoxGeometry, MeshBasicMaterial) into the scene global (if accessible) or via a temporary script tag.
**Delegation Recommendation**:
- Category: `visual-engineering` - 3D scene manipulation.
- Skills: [`frontend-ui-ux`] - Editing 3D markup/code.
**Skills Evaluation**:
- ✅ `frontend-ui-ux`: Understanding 3D/WebXR context.
**Depends On**: Task 1
**Acceptance Criteria**:
- 3D debug object added to the codebase (HTML or TS).

### Task 3: Create Robust Playwright Debug Script
**Description**:
Create `scripts/debug_screenshot.ts` to:
1. Launch browser with WebGL flags (`--use-gl=swiftshader` or `angle`, `--enable-webgl`).
2. Capture **Console Logs**, **Page Errors**, and **Network Failures** to `debug_logs.txt`.
3. Wait for:
   - DOM Overlay (`#debug-overlay`) to be visible.
   - Canvas element to exist and be larger than 0x0.
   - (Optional) "Ready" signal from app (if available).
4. Take a full-page screenshot to `debug_screenshot.png`.
**Delegation Recommendation**:
- Category: `unspecified-high` - Complex automation script.
- Skills: [`playwright`, `typescript-programmer`] - Browser automation.
**Skills Evaluation**:
- ✅ `playwright`: Core requirement.
- ✅ `typescript-programmer`: Writing the script.
**Acceptance Criteria**:
- Script `scripts/debug_screenshot.ts` exists.
- Captures logs and screenshot.
- Uses appropriate flags for headless WebGL.

### Task 4: Run Diagnostics & Analyze
**Description**:
1. Run `npx playwright test` or `node scripts/debug_screenshot.ts`.
2. Analyze `debug_screenshot.png`:
   - Is DOM overlay visible? (Yes = browser works, No = page load fail).
   - Is 3D box visible? (Yes = WebGL works, No = rendering fail).
   - Is dashboard visible?
3. Analyze `debug_logs.txt` for:
   - "Component not registered" errors.
   - 404s on GLTF/Texture assets.
   - WebGL initialization failures.
**Delegation Recommendation**:
- Category: `quick` - Running command and reading output.
- Skills: [`agent-browser`] - To view the screenshot/logs (simulated via file read).
**Depends On**: Task 2, Task 3
**Acceptance Criteria**:
- Logs and screenshot analyzed.
- Root cause identified (Headless config vs Asset missing vs App error).

### Task 5: Fix Rendering & Verify
**Description**:
Based on Task 4:
- **If Headless issue**: Update Playwright config with working flags.
- **If App Error**: Fix component registration or asset paths.
- **If Timing**: Update wait conditions (e.g., wait for specific entity).
- **Cleanup**: Remove debug overlay and 3D primitive.
- **Final**: Capture `final_screenshot.png` showing the Robot and Dashboard.
**Delegation Recommendation**:
- Category: `visual-engineering` - Code fixes.
- Skills: [`playwright`, `frontend-ui-ux`] - Verification and fixes.
**Depends On**: Task 4
**Acceptance Criteria**:
- `final_screenshot.png` shows the dashboard and robot (not just grey).
- Debug primitives removed.

## Commit Strategy
- **Commit 1**: "chore(debug): add debug overlay and primitives" (Tasks 1 & 2)
- **Commit 2**: "feat(debug): add playwright debug script" (Task 3)
- **Commit 3**: "fix(webxr): resolve screenshot rendering issue" (Task 5)
- **Commit 4**: "chore(debug): remove debug artifacts" (Task 5 cleanup)

## Success Criteria
- [ ] `scripts/debug_screenshot.ts` runs successfully (exit code 0).
- [ ] `final_screenshot.png` contains non-grey pixels (variance check or visual confirmation).
- [ ] Console logs are free of critical rendering errors.
