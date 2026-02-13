# Plan: ROS2 Visualization Update from Joint States

## Goal

Enable the ROS2 node to update the WebXR visualization with the robot's joint
states. This involves two updates:

1. **Command Visualization**: The `IKWorker` should publish the computed IK
   solution to the WebXR client (showing the *target* pose).
2. **Feedback Visualization**: The `/joint_states` subscriber should publish the
   actual robot state to the WebXR client when IK is not active (showing the
   *actual* pose).

This brings the ROS2 node's visualization capabilities in line with the `demo`
program, while handling the distributed nature of ROS2.

## Tasks

- [x] 1. Modify `IKWorker` in `teleop_xr/ros2/__main__.py`
  - Update `__init__` to accept an optional `teleop` instance (default `None`).
  - Add `set_teleop_loop(self, loop)` method.
  - Update `run()` to call `teleop.publish_joint_state` via
    `asyncio.run_coroutine_threadsafe` when a new configuration is computed.
  - Ensure thread safety and check for `teleop` and `loop` existence.
- [x] 2. Update `main()` in `teleop_xr/ros2/__main__.py`
  - Inject `teleop` instance into `ik_worker` after `teleop` is initialized.
  - Update `teleop_xr_state_callback` to capture the running asyncio loop and
    pass it to `ik_worker.set_teleop_loop`.
  - Update `joint_state_callback` to publish joint states to WebXR when IK is
    not active.
    - Access `teleop` and `loop` via the `ik_worker` instance.
    - Use `asyncio.run_coroutine_threadsafe`.
- [x] 3. Verify Implementation
  - Run `lsp_diagnostics` to ensure no errors.
  - Check imports and variable scopes (especially `ik_worker` visibility in
    `joint_state_callback`).

## Implementation Details

### IKWorker Updates

```python
class IKWorker(threading.Thread):
    def __init__(self, ..., teleop=None):
        ...
        self.teleop = teleop
        self.teleop_loop = None

    def set_teleop_loop(self, loop):
        self.teleop_loop = loop

    def run(self):
        ...
        if not np.array_equal(new_config, current_config):
            ...
            # Existing ROS2 publish
            self.publisher.publish(msg)

            # NEW: WebXR publish
            if self.teleop and self.teleop_loop:
                 joint_dict = {
                     name: float(val) for name, val in zip(
                         self.robot.actuated_joint_names, new_config
                     )
                 }
                 asyncio.run_coroutine_threadsafe(
                     self.teleop.publish_joint_state(joint_dict),
                     self.teleop_loop
                 )
```

### joint_state_callback Updates

```python
def joint_state_callback(msg: JointState):
    if not state_container.get("active", False):
        # ... existing update q logic ...

        # NEW: WebXR publish
        if ik_worker and ik_worker.teleop and ik_worker.teleop_loop:
             joint_dict = ... # extract from message or use updated q
             asyncio.run_coroutine_threadsafe(
                 ik_worker.teleop.publish_joint_state(joint_dict),
                 ik_worker.teleop_loop
             )
```

### main Loop Capture

```python
def teleop_xr_state_callback(_pose, xr_state):
    ...
    if ik_worker:
        try:
            loop = asyncio.get_running_loop()
            ik_worker.set_teleop_loop(loop)
        except RuntimeError:
            pass
    ...
```


### `joint_state_callback` Updates

```python
def joint_state_callback(msg: JointState):
    if not state_container.get("active", False):
        # ... existing update q logic ...

        # NEW: WebXR publish
        if ik_worker and ik_worker.teleop and ik_worker.teleop_loop:
             joint_dict = ... # extract from message or use updated q
             asyncio.run_coroutine_threadsafe(
                 ik_worker.teleop.publish_joint_state(joint_dict),
                 ik_worker.teleop_loop
             )
```

### `main` Loop Capture

```python
def teleop_xr_state_callback(_pose, xr_state):
    ...
    if ik_worker:
        try:
            loop = asyncio.get_running_loop()
            ik_worker.set_teleop_loop(loop)
        except RuntimeError:
            pass
    ...
```


### `joint_state_callback` Updates
```python
def joint_state_callback(msg: JointState):
    if not state_container.get("active", False):
        # ... existing update q logic ...

        # NEW: WebXR publish
        if ik_worker and ik_worker.teleop and ik_worker.teleop_loop:
             joint_dict = ... # extract from message or use updated q
             asyncio.run_coroutine_threadsafe(
                 ik_worker.teleop.publish_joint_state(joint_dict),
                 ik_worker.teleop_loop
             )
```

### `main` Loop Capture
```python
def teleop_xr_state_callback(_pose, xr_state):
    ...
    if ik_worker:
        try:
            loop = asyncio.get_running_loop()
            ik_worker.set_teleop_loop(loop)
        except RuntimeError:
            pass
    ...
```
