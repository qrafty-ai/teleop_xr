from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from teleop_xr.common_cli import CommonCLI


@dataclass
class Ros2CLI(CommonCLI):
    mode: Literal["teleop", "ik"] = "teleop"
    """Operation mode: ``teleop`` streams raw state, ``ik`` runs the IK solver."""

    head_topic: Optional[str] = None
    wrist_left_topic: Optional[str] = None
    wrist_right_topic: Optional[str] = None

    extra_streams: Dict[str, str] = field(default_factory=dict)

    frame_id: str = "xr_local"
    publish_hand_tf: bool = False

    robot_class: Optional[str] = None
    """Robot class to load (entry point or module path)."""
    robot_args: str = "{}"
    """JSON string of arguments passed to the robot constructor."""
    list_robots: bool = False
    urdf_topic: str = "/robot_description"
    urdf_timeout: float = 5.0
    no_urdf_topic: bool = False
    output_topic: str = "/joint_trajectory"

    ros_args: List[str] = field(default_factory=list)

    @property
    def robot_args_dict(self) -> Dict[str, Any]:
        try:
            return json.loads(self.robot_args)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse robot_args: {self.robot_args}")
            return {}
