# Quest3 XR State Streaming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the existing WebXR `pose` pipeline with a full `xr_state` stream (local reference space) that includes all device poses and button states, plus ROS2 publishers that map this data directly to standard ROS2 topics. Add a server-side `input_mode` option (hand/controller/auto) and remove backward-compatibility.

**Architecture:** WebXR client emits a single `xr_state` message per XR frame containing `devices[]` (head, controllers, hands). The server only accepts `xr_state`, sends a config message with `input_mode`, optionally filters devices server-side, and forwards the raw state to subscribers. ROS2 nodes convert WebXR coordinates to ROS FLU, compute relative poses (per device origin), publish PoseStamped/TF for head/controllers/hands, Joy for button states (pressed/value/touched), and PoseArray/TF for hand joints.

**Tech Stack:** WebXR (JS), FastAPI/WebSocket, Python, rclpy, geometry_msgs, sensor_msgs, tf2_ros, transforms3d.

---

### Task 1: Update WebSocket tests for `xr_state`

**Files:**
- Modify: `tests/test_pose_compounding.py`

**Step 1: Write the failing test**

Replace the pose payload helper and update the callback signature to expect a single `xr_state` dict. Capture the server `config` message in the WS on_message handler.

```python
def get_xr_state():
    return {
        "timestamp_unix_ms": 1700000000000,
        "reference_space": "local",
        "input_mode": "controller",
        "devices": [
            {
                "role": "head",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "emulated": False,
            },
            {
                "role": "controller",
                "handedness": "right",
                "targetRayPose": {
                    "position": {"x": 0.1, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "gripPose": {
                    "position": {"x": 0.1, "y": -0.1, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
                "gamepad": {
                    "buttons": [{"pressed": True, "touched": True, "value": 1.0}],
                    "axes": [0.1, -0.2],
                },
            },
        ],
    }

def callback(xr_state):
    cls.__last_message = xr_state
    cls.__callback_event.set()

def on_message(ws, message):
    data = json.loads(message)
    if data.get("type") == "config":
        cls.__last_config = data.get("data")
```

Update `_send_message` to send `type: "xr_state"`, and update tests:

```python
def _send_message(self, payload):
    message = {"type": "xr_state", "data": payload}
    self.__ws.send(json.dumps(message))

def test_config_message_on_connect(self):
    if not self.__ws_connected.is_set():
        self.skipTest("WebSocket client not connected")
    self.assertIsNotNone(self.__last_config)
    self.assertIn("input_mode", self.__last_config)

def test_response(self):
    payload = get_xr_state()
    self._send_message(payload)
    self.assertTrue(self._wait_for_callback())
    self.assertIsNotNone(self.__last_message)
    self.assertEqual(self.__last_message["reference_space"], "local")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pose_compounding.py::TestPoseCompounding::test_response -v`

Expected: FAIL because server still expects `type == "pose"` and callback signature is different.

**Step 3: Write minimal implementation**

Proceed to Task 2 to update server to accept `xr_state` and send config.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_compounding.py::TestPoseCompounding::test_response -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_pose_compounding.py
git commit -m "test: update websocket tests for xr_state"
```

---

### Task 2: Replace server `pose` handling with `xr_state`

**Files:**
- Modify: `teleop/__init__.py`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Covered by Task 1.

**Step 3: Write minimal implementation**

Update Teleop to accept `input_mode`, send config on connect, filter devices, and call callbacks with a single `xr_state` dict.

```python
def _filter_devices(xr_state, input_mode):
    if input_mode == "auto":
        return xr_state
    keep_roles = {"head"}
    if input_mode == "hand":
        keep_roles.add("hand")
    if input_mode == "controller":
        keep_roles.add("controller")
    devices = [d for d in xr_state.get("devices", []) if d.get("role") in keep_roles]
    return {**xr_state, "devices": devices}

class Teleop:
    def __init__(self, host="0.0.0.0", port=4443, input_mode="controller", **kwargs):
        self.__input_mode = input_mode
        self.__callbacks = []
        self.__manager = ConnectionManager()

    def subscribe(self, callback):
        self.__callbacks.append(callback)

    def __update(self, xr_state):
        if not isinstance(xr_state, dict):
            return
        if "devices" not in xr_state:
            return
        xr_state = _filter_devices(xr_state, self.__input_mode)
        for cb in self.__callbacks:
            cb(xr_state)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.send_personal_message({"type": "config", "data": {"input_mode": teleop.input_mode}}, websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "xr_state":
                teleop._Teleop__update(message.get("data"))
            elif message.get("type") == "log":
                logging.info(message.get("data"))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_compounding.py::TestPoseCompounding::test_response -v`

Expected: PASS

**Step 5: Commit**

```bash
git add teleop/__init__.py
git commit -m "feat: accept xr_state and send config"
```

---

### Task 3: Update WebXR client to send `xr_state` (local reference)

**Files:**
- Modify: `teleop/index.html`
- Modify: `teleop/assets/teleop-ui.js`

**Step 1: Write the failing test**

Manual-only change (no JS test harness). Add a temporary console assertion after the first frame:

```javascript
if (!xrState.devices.length) {
  console.error("xr_state contains no devices");
}
```

**Step 2: Run test to verify it fails**

Load the page and enter XR — the new structure isn’t sent yet, so the log should appear.

**Step 3: Write minimal implementation**

Replace the `pose` payload with `xr_state`, set `local` reference space, and implement device collection.

```javascript
const XR_HAND_JOINTS = [
  "wrist","thumb-metacarpal","thumb-phalanx-proximal","thumb-phalanx-distal","thumb-tip",
  "index-finger-metacarpal","index-finger-phalanx-proximal","index-finger-phalanx-intermediate","index-finger-phalanx-distal","index-finger-tip",
  "middle-finger-metacarpal","middle-finger-phalanx-proximal","middle-finger-phalanx-intermediate","middle-finger-phalanx-distal","middle-finger-tip",
  "ring-finger-metacarpal","ring-finger-phalanx-proximal","ring-finger-phalanx-intermediate","ring-finger-phalanx-distal","ring-finger-tip",
  "pinky-finger-metacarpal","pinky-finger-phalanx-proximal","pinky-finger-phalanx-intermediate","pinky-finger-phalanx-distal","pinky-finger-tip",
];

let inputMode = "auto";
const timeOriginMs = Date.now() - performance.now();

function poseToJson(pose) {
  if (!pose) return null;
  const t = pose.transform;
  return {
    position: { x: t.position.x, y: t.position.y, z: t.position.z },
    orientation: { w: t.orientation.w, x: t.orientation.x, y: t.orientation.y, z: t.orientation.z },
    emulated: pose.emulatedPosition,
    linearVelocity: pose.linearVelocity ? {
      x: pose.linearVelocity.x, y: pose.linearVelocity.y, z: pose.linearVelocity.z
    } : null,
    angularVelocity: pose.angularVelocity ? {
      x: pose.angularVelocity.x, y: pose.angularVelocity.y, z: pose.angularVelocity.z
    } : null,
  };
}

function buildXrState(frame, refSpace, session, fps) {
  const devices = [];
  const viewerPose = frame.getViewerPose(refSpace);
  if (viewerPose) {
    devices.push({
      role: "head",
      pose: poseToJson(viewerPose),
      emulated: viewerPose.emulatedPosition,
    });
  }

  for (const inputSource of session.inputSources) {
    const handedness = inputSource.handedness;
    if (inputSource.hand && inputMode !== "controller") {
      const joints = {};
      for (const jointName of XR_HAND_JOINTS) {
        const jointSpace = inputSource.hand.get(jointName);
        if (!jointSpace) continue;
        const jointPose = frame.getJointPose(jointSpace, refSpace);
        if (!jointPose) continue;
        joints[jointName] = {
          position: { x: jointPose.transform.position.x, y: jointPose.transform.position.y, z: jointPose.transform.position.z },
          orientation: { w: jointPose.transform.orientation.w, x: jointPose.transform.orientation.x, y: jointPose.transform.orientation.y, z: jointPose.transform.orientation.z },
          radius: jointPose.radius,
          emulated: jointPose.emulatedPosition,
        };
      }
      devices.push({ role: "hand", handedness, joints });
      continue;
    }

    if (!inputSource.hand && inputMode !== "hand") {
      const targetRayPose = poseToJson(frame.getPose(inputSource.targetRaySpace, refSpace));
      const gripPose = inputSource.gripSpace ? poseToJson(frame.getPose(inputSource.gripSpace, refSpace)) : null;
      const gamepad = inputSource.gamepad ? {
        buttons: inputSource.gamepad.buttons.map((b) => ({ pressed: b.pressed, touched: b.touched, value: b.value })),
        axes: Array.from(inputSource.gamepad.axes),
      } : null;
      devices.push({ role: "controller", handedness, targetRayPose, gripPose, gamepad });
    }
  }

  return {
    timestamp_unix_ms: Math.round(timeOriginMs + frame.predictedDisplayTime),
    reference_space: "local",
    input_mode: inputMode,
    devices,
    device: deviceType,
    fps,
  };
}

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "config") {
    inputMode = msg.data.input_mode || "auto";
  }
};

const session = await navigator.xr.requestSession("immersive-vr", {
  optionalFeatures: ["hand-tracking"],
});
const xrReferenceSpace = await session.requestReferenceSpace("local");

const xrState = buildXrState(frame, xrReferenceSpace, session, fps);
ws.send(JSON.stringify({ type: "xr_state", data: xrState }));
teleopUI.updateLocalStats({ position: xrState.devices[0]?.pose?.position, orientation: xrState.devices[0]?.pose?.orientation, fps });
```

If `teleop-ui.js` expects a full `state`, keep it minimal:

```javascript
updateLocalStats({ position, orientation, fps })
```

**Step 4: Run test to verify it passes**

Enter XR session and confirm the console shows `xr_state` messages with devices populated.

**Step 5: Commit**

```bash
git add teleop/index.html teleop/assets/teleop-ui.js
git commit -m "feat: send xr_state from WebXR client"
```

---

### Task 4: ROS2 XR state publishers (poses, TF, Joy, hand joints)

**Files:**
- Modify: `teleop/ros2/__main__.py`

**Step 1: Write the failing test**

Add a small unit test in `tests/test_pose_compounding.py` to validate `xr_state` filtering (server already covered). This task is mostly integration/manual verification.

**Step 2: Run test to verify it fails**

Covered by Task 1.

**Step 3: Write minimal implementation**

Add a ROS2 node that handles `xr_state` directly and publishes per-device topics.

```python
XR_HAND_JOINTS = [
    "wrist","thumb-metacarpal","thumb-phalanx-proximal","thumb-phalanx-distal","thumb-tip",
    "index-finger-metacarpal","index-finger-phalanx-proximal","index-finger-phalanx-intermediate","index-finger-phalanx-distal","index-finger-tip",
    "middle-finger-metacarpal","middle-finger-phalanx-proximal","middle-finger-phalanx-intermediate","middle-finger-phalanx-distal","middle-finger-tip",
    "ring-finger-metacarpal","ring-finger-phalanx-proximal","ring-finger-phalanx-intermediate","ring-finger-phalanx-distal","ring-finger-tip",
    "pinky-finger-metacarpal","pinky-finger-phalanx-proximal","pinky-finger-phalanx-intermediate","pinky-finger-phalanx-distal","pinky-finger-tip",
]

def pose_dict_to_matrix(pose):
    pos = pose["position"]
    quat = pose["orientation"]
    mat = t3d.affines.compose(
        [pos["x"], pos["y"], pos["z"]],
        t3d.quaternions.quat2mat([quat["w"], quat["x"], quat["y"], quat["z"]]),
        [1.0, 1.0, 1.0],
    )
    return TF_RUB2FLU @ mat

def matrix_to_pose_msg(mat):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = mat[0, 3], mat[1, 3], mat[2, 3]
    quat = t3d.quaternions.mat2quat(mat[:3, :3])
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = quat[0], quat[1], quat[2], quat[3]
    return pose

def ms_to_time(ms):
    sec = int(ms / 1000)
    nsec = int((ms - sec * 1000) * 1e6)
    return Time(seconds=sec, nanoseconds=nsec)

def build_joy(gamepad):
    if not gamepad:
        return None, None
    buttons = [1 if b.get("pressed") else 0 for b in gamepad["buttons"]]
    axes = list(gamepad.get("axes", [])) + [float(b.get("value", 0.0)) for b in gamepad["buttons"]]
    touched = [1 if b.get("touched") else 0 for b in gamepad["buttons"]]
    return (buttons, axes), touched
```

In the callback, use per-device keys and publish topics:

```python
def xr_state_callback(xr_state):
    stamp = ms_to_time(xr_state["timestamp_unix_ms"])
    for device in xr_state["devices"]:
        role = device.get("role")
        handed = device.get("handedness", "none")
        key = f"{role}_{handed}"
        if role == "head" and device.get("pose"):
            self.publish_pose("xr/head/pose", device["pose"], stamp, "xr_local", key)
        if role == "controller":
            if device.get("targetRayPose"):
                self.publish_pose(f"xr/controller_{handed}/target_ray", device["targetRayPose"], stamp, "xr_local", f"{key}_target")
            if device.get("gripPose"):
                self.publish_pose(f"xr/controller_{handed}/grip", device["gripPose"], stamp, "xr_local", f"{key}_grip")
            joy_payload, touched = build_joy(device.get("gamepad"))
            if joy_payload:
                self.publish_joy(f"xr/controller_{handed}/joy", joy_payload, stamp)
            if touched:
                self.publish_joy(f"xr/controller_{handed}/joy_touched", (touched, []), stamp)
        if role == "hand":
            self.publish_hand(device, stamp)
```

Add CLI args:

```python
parser.add_argument("--input-mode", choices=["controller", "hand", "auto"], default="controller")
parser.add_argument("--publish-hand-tf", action="store_true")
parser.add_argument("--frame-id", default="xr_local")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pose_compounding.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add teleop/ros2/__main__.py
git commit -m "feat: publish xr_state to ROS2 topics"
```

---

### Task 5: Update teleop entrypoints to new callback signature

**Files:**
- Modify: `teleop/ros2_ik/__main__.py`
- Modify: `teleop/xarm/__main__.py`
- Modify: `teleop/basic/__main__.py`
- Modify: `examples/webots/controllers/inverse_kinematics/inverse_kinematics.py`

**Step 1: Write the failing test**

Manual check only (these are not covered by pytest).

**Step 2: Run test to verify it fails**

Attempt to run each entrypoint (expect callback signature errors).

**Step 3: Write minimal implementation**

Add a shared helper in each file to pick a primary controller and extract a pose:

```python
def pick_primary_controller(xr_state, handedness="right"):
    for device in xr_state.get("devices", []):
        if device.get("role") == "controller" and device.get("handedness") == handedness:
            return device
    return None

def controller_trigger_pressed(controller):
    buttons = controller.get("gamepad", {}).get("buttons", [])
    return bool(buttons and buttons[0].get("pressed"))

def controller_gripper_value(controller):
    buttons = controller.get("gamepad", {}).get("buttons", [])
    return float(buttons[1].get("value", 0.0)) if len(buttons) > 1 else 0.0
```

Update callbacks to accept a single `xr_state` and use controller grip pose for target:

```python
def teleop_pose_callback(xr_state):
    controller = pick_primary_controller(xr_state)
    if not controller:
        return
    if not controller_trigger_pressed(controller):
        return
    pose_dict = controller.get("gripPose") or controller.get("targetRayPose")
    if not pose_dict:
        return
    pose_mat = pose_dict_to_matrix(pose_dict)
    # existing IK/servo logic using pose_mat
```

In `teleop/basic/__main__.py`, print `xr_state` directly.

**Step 4: Run test to verify it passes**

Run each entrypoint and send a sample `xr_state`:

```bash
python -m teleop.basic
python -m teleop.ros2_ik
python -m teleop.xarm
```

Expected: no callback signature errors; controller trigger gates updates.

**Step 5: Commit**

```bash
git add teleop/ros2_ik/__main__.py teleop/xarm/__main__.py teleop/basic/__main__.py examples/webots/controllers/inverse_kinematics/inverse_kinematics.py
git commit -m "refactor: adapt teleop entrypoints to xr_state"
```

---

### Task 6: Update README with schema + ROS2 topics

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

N/A (doc change).

**Step 2: Run test to verify it fails**

N/A.

**Step 3: Write minimal implementation**

Add a new section with schema and topics:

```markdown
## XR State Schema

Each frame sends:

```json
{ "type": "xr_state", "data": { "timestamp_unix_ms": 1700000000000, "reference_space": "local", "input_mode": "controller", "devices": [...] } }
```

## ROS2 Topics
- `xr/head/pose` (PoseStamped)
- `xr/controller_left/target_ray`, `xr/controller_left/grip` (PoseStamped)
- `xr/controller_left/joy`, `xr/controller_left/joy_touched` (Joy)
- `xr/hand_left/joints` (PoseArray) or TF frames if enabled
```

**Step 4: Run test to verify it passes**

N/A.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document xr_state schema and ROS2 topics"
```

---

### Task 7: Verification

**Step 1: Run automated tests**

Run: `uv run pytest -v`

Expected: PASS

**Step 2: Manual XR verification**

1. Start ROS2 node: `python -m teleop.ros2 --input-mode controller`
2. Load the WebXR page on Quest 3 and enter VR.
3. Check ROS2 topics update:
   - `ros2 topic hz /xr/controller_right/joy`
   - `ros2 topic echo /xr/controller_right/grip`

Expected: continuous updates and button changes.

**Step 3: Manual hand verification**

1. Start ROS2 node: `python -m teleop.ros2 --input-mode hand --publish-hand-tf`
2. Enable hand tracking and verify `xr/hand_left/joints` or TF frames update.

**Step 4: Commit**

No code changes; no commit required.
