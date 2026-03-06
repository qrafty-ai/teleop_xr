from typing import Literal

from pydantic import BaseModel, Field


class DeltaPose(BaseModel):
    position: dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    orientation: dict[str, float] = Field(
        default_factory=lambda: {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    )


class EEDeltaCommand(BaseModel):
    frame: Literal["left", "right", "head"] = "right"
    delta_pose: DeltaPose = Field(default_factory=DeltaPose)
