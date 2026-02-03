# Learnings: IK API Modularization

## Module Exports
- `teleop_xr.ik` now explicitly exports `BaseRobot`, `PyrokiSolver`, and `IKController`.
- Used `__all__` in `teleop_xr/ik/__init__.py` to define the public API.
- Verified exports using `uv run python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController; print('ok')"`.

## Dependency Management
- Some dependencies like `uvicorn` might be missing in the environment; using `uv sync` ensures all required packages are present.
- `uv run` is helpful to execute commands within the managed virtual environment.
