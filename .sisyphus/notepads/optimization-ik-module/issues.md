# Issues - Optimization IK Module

## Dependency Resolution

- uv add pyroki fails because it is not in the public registry. Resolved by
  adding it via its local path.

## Dependency Issues

- `teleop_xr/__init__.py` imports `uvicorn` and `fastapi` at the top level, which
  makes the package hard to use as a library without all server dependencies
  installed.
- Verification scripts must be run with `uv run` to ensure all dependencies are
  available.

## LSP False Positives with Pyroki

- LSPs like `basedpyright` incorrectly report missing arguments (like `vals`)
  for Pyroki cost functions. This is because these functions use the
  `@Cost.create_factory` decorator, which injects the `vals` argument at
  runtime, but the LSP sees the undecorated signature.
- **Workaround**: Use `# pyright: ignore[reportCallIssue]` or `# type: ignore` on
  affected lines, or suppress the check at the file level.
