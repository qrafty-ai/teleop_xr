# Learnings - Windows Path Compatibility

## Patterns & Conventions

- CI matrix strategy for multi-OS testing.
- Conditional steps in GHA based on matrix variables.

## Cross-Platform Path Normalization

- os.path.abspath returns OS-specific separators, which can cause mismatch in
  URDF string replacement on Windows.
- pathlib.Path(...).resolve().as_posix() is the preferred way to get
  forward-slash normalized paths in Python for cross-platform string matching.

### 2026-02-12

- URDF-embedded paths must use forward slashes for cross-platform compatibility
  (WebXR).
- Use `Path.as_posix()` to ensure forward slashes on Windows.
- Use `Path.resolve().as_posix()` for absolute paths in URDF.

## Windows Path Compatibility Learnings

- Using `.as_posix()` on `pathlib.Path` objects ensures that paths are embedded
  in URDFs with forward slashes, which is the standard for URDF and
  cross-platform compatibility.
- `PureWindowsPath` is useful for simulating Windows-style paths on Linux for
  unit testing.
- When testing path stripping logic, ensure that the absolute path prefix being
  stripped is also normalized to posix style, especially if the source URDF was
  generated with forward slashes.
- Mocking `pathlib.Path` in tests is necessary when simulating a different OS's
  path behavior, as `Path` is platform-dependent.

### 2026-02-12 (CI/Dependencies)

- `pin` (Pinocchio) and its dependency `cmeel-assimp` lack Windows wheels on
  PyPI, causing installation failures on Windows.
- `pyroki` is a git dependency that relies on Linux-centric robotics packages.
- core `teleop_xr` (WebXR, basic streaming) is cross-platform, but `ik` features
  are currently Linux-only.
