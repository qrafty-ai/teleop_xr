# Enable AR Passthrough & Fix Black Screen

## Context
The user wants to enable AR passthrough as the default mode and fix the "black screen" issue in VR for the A-Frame WebXR project.
- **Current State**: `webxr/index.html` has a basic `<a-scene>` with no WebXR config, no background, and no skybox.
- **Issues**:
  - VR: Black screen because no skybox/background is defined.
  - AR: Passthrough not working (or not enabled) because `xr-mode-ui` is default (VR) and background is opaque.
- **Goal**: Modify `index.html` to support AR passthrough and provide a proper VR environment.

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Configuration changes are independent |
| Task 2 | Task 1 | Validation requires the changes to be applied |

## Parallel Execution Graph

Wave 1 (Start immediately):
└── Task 1: Configure webxr/index.html for AR and VR (no dependencies)

Wave 2 (After Wave 1):
└── Task 2: Verify Configuration & Render (depends: Task 1)

## Tasks

### Task 1: Configure webxr/index.html for AR and VR
**Description**:
Modify `webxr/index.html` to:
1.  Update `<a-scene>` attributes:
    - `xr-mode-ui="XRMode: ar"` (Prioritize AR entry)
    - `webxr="optionalFeatures: hit-test, dom-overlay; overlayElement: #overlay"`
    - `background="transparent: true"` (Critical for AR passthrough)
2.  Add a skybox for VR that hides in AR:
    - `<a-sky color="#ECECEC" hide-on-enter-ar></a-sky>`
3.  Add the overlay container (empty div) to satisfy `overlayElement`:
    - `<div id="overlay"></div>` (inside body, outside scene or as sibling)

**Delegation Recommendation**:
- Category: `quick` - Simple HTML modification
- Skills: [`frontend-ui-ux`] - HTML editing

**Skills Evaluation**:
- INCLUDED `frontend-ui-ux`: For correct HTML structure edits.
- OMITTED `typescript-programmer`: Not touching TS files.

**Acceptance Criteria**:
- `webxr/index.html` contains `xr-mode-ui="XRMode: ar"`.
- `webxr/index.html` contains `background="transparent: true"`.
- `webxr/index.html` contains `<a-sky` with `hide-on-enter-ar`.
- `webxr/index.html` contains `#overlay` element.

### Task 2: Verify Configuration & Render
**Description**:
Verify the changes using a static analysis script (Metis-recommended) and a browser screenshot of the 2D fallback.

**Delegation Recommendation**:
- Category: `quick` - Verification task
- Skills: [`agent-browser`] - For screenshot and node script execution

**Skills Evaluation**:
- INCLUDED `agent-browser`: To take a screenshot of the running app.

**Acceptance Criteria**:
- **Static Analysis**: Run this Node script and ensure output is "ok":
  ```bash
  node -e "const fs=require('fs'); const html=fs.readFileSync('webxr/index.html','utf8'); const checks=[[/<a-scene[^>]*xr-mode-ui=/, 'missing xr-mode-ui'], [/XRMode:\s*ar/, 'XRMode must be ar'], [/transparent:\s*true/, 'missing background transparent'], [/hide-on-enter-ar/, 'missing hide-on-enter-ar skybox']]; for(const [re,msg] of checks){if(!re.test(html)){console.error('FAIL:',msg);process.exit(1)}} console.log('ok');"
  ```
- **Visual Check**: Screenshot of `http://localhost:5173` (or configured port) saved to `.sisyphus/evidence/ar_fix.png`.

## Commit Strategy
- Commit 1: `feat(webxr): enable AR passthrough and add VR skybox` (after Task 1 & 2 verified)

## Success Criteria
- Static analysis passes (all attributes present).
- App builds and loads without console errors in browser.
