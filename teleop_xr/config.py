from enum import Enum
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field


class InputMode(str, Enum):
    CONTROLLER = "controller"
    HAND = "hand"
    AUTO = "auto"


class ViewConfig(BaseModel):
    device: Union[int, str]
    width: int = 1280
    height: int = 720
    fps: int = 30


class RobotVisConfig(BaseModel):
    urdf_path: str
    mesh_path: Optional[str] = None
    model_scale: float = 1.0
    initial_rotation_euler: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class TeleopSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 4443
    robot_vis: Optional[RobotVisConfig] = None
    input_mode: InputMode = InputMode.CONTROLLER
    natural_phone_orientation_euler: List[float] = Field(
        default_factory=lambda: [0.0, -0.7853981633974483, 0.0]
    )
    natural_phone_position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    camera_views: Dict[str, ViewConfig] = Field(default_factory=dict)
    video_config: Optional[Dict[str, Any]] = None
