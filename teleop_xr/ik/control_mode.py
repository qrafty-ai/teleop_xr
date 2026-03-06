from enum import Enum


class ControlMode(str, Enum):
    TELEOP = "teleop"
    EE_DELTA = "ee_delta"
