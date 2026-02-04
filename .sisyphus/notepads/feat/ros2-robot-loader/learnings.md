- Fixed RAM xacro patch signature mismatch.
- Implemented state caching in Teleop for immediate client sync.
- Updated Franka default pose to a non-singular home position.
- Removed local URDF assets in favor of RAM-fetched descriptions.

- Encountered issues with pytest-cov missing in default environment; resolved by using `uv run pytest` which uses the project's managed dependencies.
- Pre-commit hooks auto-fixed formatting and linting issues in the new test files (ruff E701, unused imports, trailing whitespace).
