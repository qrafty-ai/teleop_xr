# Design Plan: Tyro CLI & Pydantic Configuration/Messaging

## 1. Core Architecture

We will introduce two new modules: `teleop_xr.config` for static configuration and `teleop_xr.messages` for WebSocket protocols.

### Dependencies
- **Add**: `tyro` (for CLI), `pydantic` (for validation).

### Configuration Models (`teleop_xr/config.py`)

1.  **`InputMode` (Enum)**: `CONTROLLER`, `HAND`, `AUTO`.
2.  **`ViewConfig`**: Describes a camera view (device ID, resolution, etc.).
3.  **`TeleopSettings`**: The initialization config sent to WebXR.
    *   `input_mode`: `InputMode`
    *   `camera_views`: `Dict[str, ViewConfig]`
    *   `video_config`: `VideoConfig`

### Message Models (`teleop_xr/messages.py`)

We will define the WebSocket protocol using Pydantic models.

1.  **`XRDeviceRole` (Enum)**: `head`, `controller`, `hand`.
2.  **`XRHandedness` (Enum)**: `left`, `right`, `none`.
3.  **`XRButton`**:
    *   `pressed`: bool
    *   `touched`: bool
    *   `value`: float (0.0 - 1.0)
4.  **`XRGamepad`**:
    *   `buttons`: List[`XRButton`]
    *   `axes`: List[float]
5.  **`XRInputSource`**:
    *   `role`: `XRDeviceRole`
    *   `handedness`: `XRHandedness`
    *   `pose`: Optional[`XRPose`]
    *   `gripPose`: Optional[`XRPose`]
    *   `gamepad`: Optional[`XRGamepad`] (Added support for joystick/buttons)
    *   `joints`: Optional[Dict[str, `XRPose`]]
6.  **`XRStateMessage`**:
    *   `type`: Literal["xr_state"]
    *   `data`: Dict containing `devices` list, `fetch_latency_ms`, etc.

**Performance Strategy**:
- **Outbound (Python -> JS)**: Use `model.model_dump_json()` to ensure valid JSON is sent for config/setup.
- **Inbound (JS -> Python)**:
    - For `config/setup` messages: Use `Model.model_validate_json(data)`.
    - For `xr_state` (High Freq): We will **NOT** run full Pydantic validation on every frame in the hot loop.
    - Instead, we will parse `json.loads(data)` and access the dict directly (as current code does) for minimal latency.
    - The Pydantic model serves as the **schema definition** and documentation.

## 2. CLI Refactoring with Tyro

We'll replace `argparse` with `tyro` dataclasses.

### Base Configuration
```python
@dataclass
class CommonCLI:
    host: str = "0.0.0.0"
    port: int = 4433
    input_mode: InputMode = InputMode.CONTROLLER
```

### Basic Interface (`basic/__main__.py`)
```python
@dataclass
class BasicCLI(CommonCLI):
    # Tyro handles Union[int, str] correctly for device IDs/paths
    head_device: Union[int, str, None] = None
    wrist_left_device: Union[int, str, None] = None
    wrist_right_device: Union[int, str, None] = None
```

### ROS2 Interface (`ros2/__main__.py`)
```python
@dataclass
class Ros2CLI(CommonCLI):
    frame_id: str = "xr_local"
    publish_hand_tf: bool = False

    # Explicit topic arguments for standard views
    head_topic: Optional[str] = None
    wrist_left_topic: Optional[str] = None
    wrist_right_topic: Optional[str] = None

    # Additional streams: --extra-streams gripper=/gripper/camera
    extra_streams: Dict[str, str] = field(default_factory=dict)
```

## 3. Integration Plan

1.  **Dependencies**: Add `tyro` and `pydantic`.
2.  **Define Models**: Create `teleop_xr/config.py` and `teleop_xr/messages.py`.
3.  **Update `Teleop` Class**:
    *   Init accepts `TeleopSettings`.
    *   `run()` sends `TeleopSettings.model_dump_json()`.
4.  **Refactor CLI**:
    *   Update `basic/__main__.py` to use `tyro.cli(BasicCLI)`.
    *   Update `ros2/__main__.py` to use `tyro.cli(Ros2CLI)`.
    *   Ensure `ros_args` are handled correctly for `rclpy`.
5.  **Verification**: Ensure `json.loads` is still used in the `while True` loop for `xr_state` to satisfy the "no validation every time" requirement.
