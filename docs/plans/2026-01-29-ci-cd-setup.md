# GitHub Actions & Versioning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up a robust CI/CD pipeline using GitHub Actions that enforces code quality (pre-commit, tests) and automates releases (build, publish) based on Git tags using `hatch-vcs`.

**Architecture:**
- **Versioning:** Switch from static `version = "..."` to dynamic `hatch-vcs` (Git tags).
- **CI/CD:** A single GitHub workflow (`ci.yml`) handling:
    1. **Quality:** `pre-commit` checks.
    2. **Testing:** `pytest` execution.
    3. **Build:** `uv build` (Python + WebXR).
    4. **Publish:** `uv publish` to PyPI (Trusted Publishing) triggered by tags.

**Tech Stack:** GitHub Actions, `uv`, `hatchling`, `hatch-vcs`, PyPI Trusted Publishing.

---

### Task 1: Configure Dynamic Versioning

**Files:**
- Modify: `pyproject.toml`
- Modify: `teleop_xr/__init__.py` (optional verify)

**Step 1: Update `pyproject.toml`**
Remove static version and configure `hatch-vcs`.

```toml
[build-system]
requires = ["hatchling", "hatch-vcs", "hatch-build-scripts"]
build-backend = "hatchling.build"

[project]
name = "teleop_xr"
dynamic = ["version"]
# REMOVE: version = "..."

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "teleop_xr/_version.py"
```

**Step 2: Add `_version.py` handling in `__init__.py`**
Allow the package to know its own version at runtime.

```python
try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
```

**Step 3: Verify local build**
1. Tag current commit (e.g., `git tag v0.1.2-dev`).
2. Run `uv build`.
3. Check `dist/` filename has correct version.

**Step 4: Commit**
```bash
git add pyproject.toml teleop_xr/__init__.py
git commit -m "build: switch to dynamic versioning with hatch-vcs"
```

---

### Task 2: Create GitHub Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Define Workflow Structure**

Create `.github/workflows/ci.yml` with the following content:

```yaml
name: CI/CD

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

permissions:
  contents: read

jobs:
  quality:
    name: Check Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"
      - name: Install dependencies
        run: uv sync --all-extras --dev
      - name: Run pre-commit
        run: uv run pre-commit run --all-files

  tests:
    name: Run Tests
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"
      - name: Install dependencies
        run: uv sync --all-extras --dev
      - name: Run pytest
        run: uv run pytest

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: tests
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Required for hatch-vcs to read tags
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: "pyproject.toml"
      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
      - name: Build
        run: uv build
      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: build
    if: startsWith(github.ref, 'refs/tags/v')  # Only publish on tags
    environment:
      name: pypi
      url: https://pypi.org/p/teleop_xr
    permissions:
      id-token: write  # Required for Trusted Publishing
    steps:
      - name: Download Artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Publish
        run: uv publish
```

**Step 2: Commit**
```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for quality, test, build, and publish"
```

---

### Task 3: Final Verification

**Step 1: Verify `uv build` in clean environment**
Ensure local `uv build` still works with the new `hatch-vcs` configuration.

**Step 2: Tag and Push (Simulate)**
(User action required later) Explain how to push a tag to trigger release.
