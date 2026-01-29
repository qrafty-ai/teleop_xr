# Camera Streaming Refactor Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decouple camera opening from the `teleop` library to allow external video sources (e.g., ROS2 topics) to feed the video stream, while maintaining support for the existing internal OpenCV camera opening.

**Architecture:**
- Introduce a `VideoSource` protocol/interface to abstract video frame retrieval.
- Split existing `ThreadedVideoCapture` into `OpenCVVideoSource` (implementation of `VideoSource`).
- Update `CameraStreamTrack` to consume `VideoSource` instead of managing `cv2.VideoCapture` directly.
- Refactor `Teleop` class to accept a registry of `VideoSource` instances.
- In `teleop/ros2/__main__.py`, implement a bridge between ROS2 image topics and `VideoSource`.

**Tech Stack:** Python, aiortc, OpenCV, ROS2 (rclpy)

---

### Task 1: Environment Setup

**Files:**
- Modify: `.worktrees/camera-streaming/setup.sh` (or just run commands)

**Step 1: Create virtual environment and install dependencies**
(We need to ensure we can run tests and code in the worktree)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[utils] pytest
```

---

### Task 2: Define VideoSource Interface and OpenCV Implementation

**Files:**
- Modify: `teleop/video_stream.py`

**Step 1: Define VideoSource Protocol**
Create an abstract base class or Protocol `VideoSource` with methods:
- `start()`
- `stop()`
- `read() -> (bool, np.ndarray | None)`
- `properties` like width, height, fps (optional but helpful)

**Step 2: Rename and Refactor ThreadedVideoCapture**
Rename `ThreadedVideoCapture` to `OpenCVVideoSource`.
Ensure it implements `VideoSource`.
Preserve all existing logic (threading, V4L2 usage).

**Step 3: Create ExternalVideoSource**
Create a simple implementation `ExternalVideoSource` that allows pushing frames from outside.
Methods:
- `put_frame(frame: np.ndarray)`
- `read() -> (bool, frame)`

**Step 4: Update CameraStreamTrack**
Modify `CameraStreamTrack.__init__` to accept a `VideoSource` instance instead of `VideoStreamConfig` (or in addition to).
Remove `_ensure_capture` logic that instantiates `ThreadedVideoCapture`.
Instead, use the provided `VideoSource`.

**Step 5: Test Refactor**
Run existing tests (if any) or create a simple test script to verify `OpenCVVideoSource` works.

---

### Task 3: Refactor VideoStreamManager and Teleop Class

**Files:**
- Modify: `teleop/video_stream.py`
- Modify: `teleop/__init__.py`

**Step 1: Update VideoStreamManager**
Update `VideoStreamManager.__init__` to accept a dictionary mapping `stream_id` to `VideoSource` instances.
Update `create_offer` to use these sources when building tracks.

**Step 2: Update Teleop Class initialization**
Modify `Teleop.__init__` to accept `video_sources: dict[str, VideoSource] = None`.
If `video_sources` is provided, use them.
If not, maintain backward compatibility: use `camera_views` config to create `OpenCVVideoSource` instances automatically.

**Step 3: Update Teleop._start_video_session**
Pass the resolved `video_sources` to `VideoStreamManager`.

---

### Task 4: ROS2 Image Bridge

**Files:**
- Modify: `teleop/ros2/__main__.py`

**Step 1: Create ROS2 Video Source Helper**
Create a class `ROSImageVideoSource(ExternalVideoSource)` or just a helper function.
It should subscribe to a ROS image topic.
In the callback, convert ROS Image to OpenCV/numpy format and call `put_frame`.

**Step 2: Update main()**
Parse arguments for video sources (e.g., `--camera-stream head=/camera/image_raw`).
Create `ROSImageVideoSource` instances for requested streams.
Pass these sources to `Teleop` constructor.

**Step 3: Verification**
Verify that `rclpy.spin_once` in the loop drives the image callbacks sufficiently.

---
