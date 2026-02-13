# Decisions - ROS2 Parameter URDF Refactor

## Standardized Robot Constructor

- Decision: Standardize robot subclass constructors to
  `__init__(self, urdf_string: str | None = None, **kwargs: Any)`.
- Rationale: Ensures a consistent instantiation pattern for all robots, whether
  loaded by default, from a string override, or with additional custom
  arguments. Enables simpler factory/loader logic.

## Centralized URDF/Mesh Management

- Decision: Move `urdf_path` and `mesh_path` state into `BaseRobot` and drive
  them via `_load_urdf`.
- Rationale: Reduces duplication and ensures that visualization
  (`get_vis_config`) always has access to the paths of the actually loaded
  model.

## Scale as a Property

- Decision: Use a `model_scale` property in the base class that defaults to 1.0.
- Rationale: Allows subclasses to easily override the visualization scale
  without having to reimplement the entire `get_vis_config` logic.
