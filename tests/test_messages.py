from typing import cast
import pytest
from pydantic import ValidationError
from teleop_xr.messages import (
    XRButtonState,
    XRGamepad,
    XRHandedness,
    XRDeviceRole,
    XRPose,
    XRInputSource,
    XRState,
    XRStateMessage,
    XRButtonEvent,
)


def test_xr_button_state():
    btn = XRButtonState(pressed=True, touched=False, value=1.0)
    assert btn.pressed is True
    assert btn.touched is False
    assert btn.value == 1.0

    with pytest.raises(ValidationError):
        XRButtonState(pressed=cast(bool, "not_bool"), touched=False, value=1.0)


def test_xr_gamepad():
    btn1 = XRButtonState(pressed=True, touched=False, value=1.0)
    btn2 = XRButtonState(pressed=False, touched=True, value=0.0)
    gamepad = XRGamepad(buttons=[btn1, btn2], axes=[0.5, -0.5])

    assert len(gamepad.buttons) == 2
    assert len(gamepad.axes) == 2
    assert gamepad.buttons[0].pressed is True


def test_xr_enums():
    assert XRHandedness.LEFT == "left"
    assert XRHandedness.RIGHT == "right"
    assert XRHandedness.NONE == "none"

    assert XRDeviceRole.HEAD == "head"
    assert XRDeviceRole.CONTROLLER == "controller"
    assert XRDeviceRole.HAND == "hand"


def test_xr_pose():
    pose = XRPose(
        position={"x": 1.0, "y": 2.0, "z": 3.0},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )
    assert pose.position["x"] == 1.0
    assert pose.orientation["w"] == 1.0

    with pytest.raises(ValidationError):
        XRPose(**{"position": {"x": 1.0}})


def test_xr_input_source():
    pose = XRPose(
        position={"x": 1.0, "y": 2.0, "z": 3.0},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )
    source = XRInputSource(
        role=XRDeviceRole.CONTROLLER, handedness=XRHandedness.LEFT, pose=pose
    )
    assert source.role == XRDeviceRole.CONTROLLER
    assert source.handedness == XRHandedness.LEFT
    assert source.pose == pose
    assert source.gripPose is None
    assert source.gamepad is None


def test_xr_state():
    pose = XRPose(
        position={"x": 1.0, "y": 2.0, "z": 3.0},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )
    source = XRInputSource(role=XRDeviceRole.HEAD, pose=pose)
    state = XRState(
        timestamp_unix_ms=1616161616.0, devices=[source], fetch_latency_ms=10.0
    )
    assert state.timestamp_unix_ms == 1616161616.0
    assert len(state.devices) == 1
    assert state.fetch_latency_ms == 10.0


def test_xr_state_message():
    pose = XRPose(
        position={"x": 1.0, "y": 2.0, "z": 3.0},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )
    source = XRInputSource(role=XRDeviceRole.HEAD, pose=pose)
    state = XRState(timestamp_unix_ms=1000.0, devices=[source])

    msg = XRStateMessage(data=state)
    assert msg.type == "xr_state"
    assert msg.data == state


def test_xr_button_event():
    event = XRButtonEvent(
        type="press",
        button="trigger",
        controller="left",
        timestamp_ms=2000.0,
        hold_duration_ms=100.0,
    )
    assert event.type == "press"
    assert event.button == "trigger"
    assert event.controller == "left"
    assert event.timestamp_ms == 2000.0
    assert event.hold_duration_ms == 100.0

    event_no_hold = XRButtonEvent(
        type="release", button="grip", controller="right", timestamp_ms=3000.0
    )
    assert event_no_hold.hold_duration_ms is None
