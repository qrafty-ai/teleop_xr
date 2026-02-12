### Decided Architecture
- Run tests on both Ubuntu and Windows to ensure cross-platform compatibility.
- Only upload coverage from Ubuntu to avoid duplication and simplify reporting.
## Robot Visualization Path Normalization
- Normalized mesh path stripping to use forward slashes via pathlib.Path.as_posix() to ensure cross-platform compatibility with URDF content.
### 2026-02-12
- Modified `ram.py` to use `.as_posix()` for all URDF-embedded paths in `_replace_package_uris` and `_replace_dae_with_glb`.

### 2026-02-12 (CI/Dependencies)
- Moved `pin` from `dev` group to `ik` extra in `pyproject.toml` to allow dev environment setup on Windows without failing on missing wheels.
- Excluded `ik` extra from Windows CI in `.github/workflows/ci.yml`.
- Added `pytest.importorskip("pyroki")` to IK-dependent tests to handle missing dependencies gracefully on Windows.
