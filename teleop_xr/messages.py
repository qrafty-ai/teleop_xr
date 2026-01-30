from enum import Enum
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel


class XRButtonState(BaseModel):
    pressed: bool
    touched: bool
    value: float


class XRGamepad(BaseModel):
    buttons: List[XRButtonState]
    axes: List[float]


class XRHandedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class XRDeviceRole(str, Enum):
    HEAD = "head"
    CONTROLLER = "controller"
    HAND = "hand"


class XRPose(BaseModel):
    position: Dict[str, float]
    orientation: Dict[str, float]


class XRInputSource(BaseModel):
    role: XRDeviceRole
    handedness: XRHandedness = XRHandedness.NONE
    pose: Optional[XRPose] = None
    gripPose: Optional[XRPose] = None
    gamepad: Optional[XRGamepad] = None
    joints: Optional[Dict[str, XRPose]] = None


class XRState(BaseModel):
    timestamp_unix_ms: float
    fetch_latency_ms: Optional[float] = None
    devices: List[XRInputSource]


class XRStateMessage(BaseModel):
    type: Literal["xr_state"] = "xr_state"
    data: XRState


class XRButtonEvent(BaseModel):
    type: str
    button: str
    controller: str
    timestamp_ms: float
    hold_duration_ms: Optional[float] = None
