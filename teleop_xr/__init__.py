import os
import math
import socket
import logging
from typing import Callable, List, Optional, Dict, Union, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import transforms3d as t3d
import numpy as np
import json

from teleop_xr.video_stream import (
    VideoStreamConfig,
    VideoStreamManager,
    parse_video_config,
    route_video_message,
    VideoSource,
    build_sources,
)
from teleop_xr.camera_views import build_video_streams
from teleop_xr.config import TeleopSettings

TF_RUB2FLU = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


def _resolve_frontend_paths(package_dir: str) -> tuple[str, str, str, str]:
    dist_dir = os.path.join(package_dir, "dist")
    dist_index = os.path.join(dist_dir, "index.html")

    if not os.path.exists(dist_index):
        # Fallback for development (repo root)
        repo_root = os.path.abspath(os.path.join(package_dir, os.pardir))
        dist_dir = os.path.join(repo_root, "webxr", "dist")
        dist_index = os.path.join(dist_dir, "index.html")

    return dist_dir, dist_index, "/", "webxr"


def get_local_ip():
    try:
        # Connect to an external address (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS as a dummy target
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error: {e}"


def are_close(a, b=None, lin_tol=1e-9, ang_tol=1e-9):
    """
    Check if two transformation matrices are close to each other within specified tolerances.

    Parameters:
        a (numpy.ndarray): The first transformation matrix.
        b (numpy.ndarray, optional): The second transformation matrix. If not provided, it defaults to the identity matrix.
        lin_tol (float, optional): The linear tolerance for closeness. Defaults to 1e-9.
        ang_tol (float, optional): The angular tolerance for closeness. Defaults to 1e-9.

    Returns:
        bool: True if the matrices are close, False otherwise.
    """
    if b is None:
        b = np.eye(4)
    d = np.linalg.inv(a) @ b
    if not np.allclose(d[:3, 3], np.zeros(3), atol=lin_tol):
        return False
    yaw = math.atan2(d[1, 0], d[0, 0])
    pitch = math.asin(-d[2, 0])
    roll = math.atan2(d[2, 1], d[2, 2])
    rpy = np.array([roll, pitch, yaw])
    return np.allclose(rpy, np.zeros(3), atol=ang_tol)


def slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    # If the dot product is negative, use the shortest path
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Linear interpolation fallback for nearly identical quaternions
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)

    return q1 * np.cos(theta) + q3 * np.sin(theta)


def interpolate_transforms(T1, T2, alpha):
    """
    Interpolate between two 4x4 transformation matrices using SLERP + linear translation.

    Args:
        T1 (np.ndarray): Start transform (4x4)
        T2 (np.ndarray): End transform (4x4)
        alpha (float): Interpolation factor [0, 1]

    Returns:
        np.ndarray: Interpolated transform (4x4)
    """
    assert T1.shape == (4, 4) and T2.shape == (4, 4)
    assert 0.0 <= alpha <= 1.0

    # Translation
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    t_interp = (1 - alpha) * t1 + alpha * t2

    # Rotation
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    q1 = t3d.quaternions.mat2quat(R1)
    q2 = t3d.quaternions.mat2quat(R2)

    # SLERP
    q_interp = slerp(q1, q2, alpha)
    R_interp = t3d.quaternions.quat2mat(q_interp)

    # Final transform
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp

    return T_interp


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Remove broken connections
                self.active_connections.remove(connection)


class Teleop:
    """
    Teleop class for controlling a robot remotely using FastAPI and WebSockets.

    Args:
        settings (TeleopSettings): Configuration settings for the teleop server.
        video_sources (Optional[dict[str, VideoSource]]): Dictionary of video sources.
    """

    def __init__(
        self,
        settings: TeleopSettings,
        video_sources: Optional[dict[str, VideoSource]] = None,
    ):
        self.__logger = logging.getLogger("teleop")
        self.__logger.setLevel(logging.INFO)
        if not self.__logger.handlers:
            self.__logger.addHandler(logging.StreamHandler())

        self.__settings = settings
        self.__camera_views = self.__settings.camera_views

        self.__multi_eef_mode = getattr(self.__settings, "multi_eef_mode", False)
        self.__relative_pose_init: Dict[str, Optional[np.ndarray]] = {}
        self.__absolute_pose_init: Dict[str, Optional[np.ndarray]] = {}
        self.__previous_received_pose: Dict[str, Optional[np.ndarray]] = {}

        if self.__multi_eef_mode:
            self.__pose: Union[np.ndarray, Dict[str, np.ndarray]] = {
                "head": np.eye(4),
                "left_hand": np.eye(4),
                "right_hand": np.eye(4),
            }
        else:
            self.__pose: Union[np.ndarray, Dict[str, np.ndarray]] = np.eye(4)

        self.__callbacks = []

        euler = self.__settings.natural_phone_orientation_euler
        self.__natural_phone_pose = t3d.affines.compose(
            self.__settings.natural_phone_position,
            t3d.euler.euler2mat(euler[0], euler[1], euler[2]),
            [1, 1, 1],
        )

        self.__app = FastAPI()
        self.__manager = ConnectionManager()

        self.__video_streams: list[VideoStreamConfig] = []
        self.__video_sources: dict[str, VideoSource] = video_sources or {}
        self.__video_sessions: dict[WebSocket, VideoStreamManager] = {}

        if self.__camera_views is not None and not self.__video_streams:
            self.set_video_streams(build_video_streams(self.__camera_views))

        # Configure logging
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        self.__setup_routes()

    @property
    def input_mode(self):
        return self.__settings.input_mode

    def set_pose(self, pose: np.ndarray) -> None:
        """
        Set the current pose of the end-effector.

        Parameters:
        - pose (np.ndarray): A 4x4 transformation matrix representing the pose.
        """
        if isinstance(self.__pose, dict):
            self.__pose["right_hand"] = pose
        else:
            self.__pose = pose

    def subscribe(self, callback: Callable[[Any, Dict[Any, Any]], None]) -> None:
        """
        Subscribe to receive updates from the teleop module.

        Parameters:
            callback (Callable[[Any, dict], None]): A callback function that will be called when pose updates are received.
                The callback function should take two arguments:
                    - np.ndarray or Dict[str, np.ndarray]: Transformation(s) representing the target pose(s).
                    - dict: A dictionary containing additional information.
        """
        self.__callbacks.append(callback)

    def __notify_subscribers(self, pose, message):
        for callback in self.__callbacks:
            callback(pose, message)

    def set_video_streams(self, payload: Dict[Any, Any]) -> None:
        self.__video_streams = parse_video_config(payload)

    def clear_video_streams(self) -> None:
        self.__video_streams = []

    async def _start_video_session(self, websocket: WebSocket) -> None:
        sources = self.__video_sources
        if not sources and self.__video_streams:
            sources = build_sources(self.__video_streams)

        manager = VideoStreamManager(sources)
        self.__video_sessions[websocket] = manager
        offer = await manager.create_offer()
        await self.__manager.send_personal_message(
            json.dumps(
                {"type": "video_offer", "data": {"sdp": offer.sdp, "type": offer.type}}
            ),
            websocket,
        )

    async def _handle_video_message(
        self, websocket: WebSocket, message: Dict[Any, Any]
    ) -> None:
        msg_type = message.get("type")
        self.__logger.info(f"Received video message: {msg_type}")

        if msg_type == "video_request":
            self.__logger.info("Starting video session")
            await self._start_video_session(websocket)
            return

        session = self.__video_sessions.get(websocket)
        if session:
            self.__logger.debug(f"Routing {msg_type} to session")
            await route_video_message(session, message)
        else:
            self.__logger.warning(
                f"No active video session for websocket, ignoring {msg_type}"
            )

    def __process_pose(
        self, position, orientation, target_key="default", move=True, scale=1.0
    ):
        position_arr = np.array([position["x"], position["y"], position["z"]])
        quat = np.array(
            [orientation["w"], orientation["x"], orientation["y"], orientation["z"]]
        )

        rel_init = self.__relative_pose_init.get(target_key)
        abs_init = self.__absolute_pose_init.get(target_key)
        prev_recv = self.__previous_received_pose.get(target_key)

        if isinstance(self.__pose, dict):
            base_pose = self.__pose.get(target_key, np.eye(4))
        else:
            base_pose = self.__pose

        if not move:
            self.__relative_pose_init[target_key] = None
            self.__absolute_pose_init[target_key] = None
            return base_pose

        received_pose_rub = t3d.affines.compose(
            position_arr, t3d.quaternions.quat2mat(quat), [1, 1, 1]
        )
        received_pose = TF_RUB2FLU @ received_pose_rub
        received_pose[:3, :3] = received_pose[:3, :3] @ np.linalg.inv(
            TF_RUB2FLU[:3, :3]
        )
        received_pose = received_pose @ self.__natural_phone_pose

        if prev_recv is not None:
            if not are_close(
                received_pose,
                prev_recv,
                lin_tol=0.05,
                ang_tol=math.radians(35),
            ):
                self.__logger.warning(
                    f"Pose jump detected for {target_key}, resetting the pose"
                )
                self.__relative_pose_init[target_key] = None
                self.__previous_received_pose[target_key] = received_pose
                return base_pose

        self.__previous_received_pose[target_key] = received_pose

        if rel_init is None:
            rel_init = received_pose
            abs_init = base_pose
            self.__relative_pose_init[target_key] = rel_init
            self.__absolute_pose_init[target_key] = abs_init
            self.__previous_received_pose[target_key] = None

        assert abs_init is not None
        relative_position = received_pose[:3, 3] - rel_init[:3, 3]
        relative_orientation = received_pose[:3, :3] @ np.linalg.inv(rel_init[:3, :3])

        if scale > 1.0:
            relative_position *= scale

        new_pose = np.eye(4)
        new_pose[:3, 3] = abs_init[:3, 3] + relative_position
        new_pose[:3, :3] = relative_orientation @ abs_init[:3, :3]

        if scale < 1.0:
            new_pose = interpolate_transforms(abs_init, new_pose, scale)

        return new_pose

    def apply(
        self,
        position=None,
        orientation=None,
        move=True,
        scale=1.0,
        info=None,
        targets=None,
    ):
        if info is None:
            info = {}

        if not self.__multi_eef_mode:
            if position is not None and orientation is not None:
                self.__pose = self.__process_pose(
                    position, orientation, target_key="default", move=move, scale=scale
                )
            self.__notify_subscribers(self.__pose, info)
        else:
            if targets:
                for target_key, data in targets.items():
                    if target_key not in ["head", "left_hand", "right_hand"]:
                        continue
                    if isinstance(self.__pose, dict):
                        self.__pose[target_key] = self.__process_pose(
                            data["position"],
                            data["orientation"],
                            target_key=target_key,
                            move=move,
                            scale=scale,
                        )
            elif position is not None and orientation is not None:
                # Fallback for single pose call in multi-mode, update right_hand by default
                if isinstance(self.__pose, dict):
                    self.__pose["right_hand"] = self.__process_pose(
                        position,
                        orientation,
                        target_key="right_hand",
                        move=move,
                        scale=scale,
                    )

            self.__notify_subscribers(self.__pose, info)

    def __handle_xr_state(self, message):
        input_mode = self.__settings.input_mode
        devices = message.get("devices", [])

        # Log fetch latency if present
        fetch_latency = message.get("fetch_latency_ms")
        if fetch_latency is not None:
            self.__logger.debug(f"XR input fetch latency: {fetch_latency:.2f}ms")

        if self.__multi_eef_mode:
            targets = {}
            for d in devices:
                role = d.get("role")
                if role == "head":
                    pose_data = d.get("pose")
                    if pose_data:
                        targets["head"] = pose_data
                elif role == "controller" or role == "hand":
                    handedness = d.get("handedness")
                    target_key = "left_hand" if handedness == "left" else "right_hand"

                    # Prefer gripPose for controllers, pose for hands
                    pose_data = d.get("gripPose") or d.get("pose")
                    if pose_data:
                        targets[target_key] = pose_data

            self.apply(move=True, scale=1.0, info=message, targets=targets)
            return

        filtered_devices = []
        for d in devices:
            role = d.get("role")
            if role == "head":
                filtered_devices.append(d)
            elif role == "controller" and input_mode in ["controller", "auto"]:
                filtered_devices.append(d)

        message["devices"] = filtered_devices

        # Derive pose from controller device (prefer right then left)
        target_device = None
        for handedness in ["right", "left"]:
            for d in filtered_devices:
                if d.get("role") == "controller" and d.get("handedness") == handedness:
                    target_device = d
                    break
            if target_device:
                break

        if target_device:
            pose_data = target_device.get("gripPose")
            if pose_data:
                self.apply(
                    pose_data["position"],
                    pose_data["orientation"],
                    move=True,
                    scale=1.0,
                    info=message,
                )
                return

        # Fallback to head pose
        for d in filtered_devices:
            if d.get("role") == "head":
                pose_data = d.get("pose")
                if pose_data:
                    self.apply(
                        pose_data["position"],
                        pose_data["orientation"],
                        move=True,
                        scale=1.0,
                        info=message,
                    )
                    return

    def __setup_routes(self):
        static_dir, index_path, mount_path, mount_name = _resolve_frontend_paths(
            THIS_DIR
        )

        @self.__app.get("/")
        async def index():
            return FileResponse(index_path)

        @self.__app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.__manager.connect(websocket)
            self.__logger.info("Client connected")

            # Send config message on connect
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "config",
                        "data": self.__settings.model_dump(),
                    }
                )
            )

            if self.__video_streams:
                await self.__manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "video_config",
                            "data": {
                                "streams": [s.__dict__ for s in self.__video_streams]
                            },
                        }
                    ),
                    websocket,
                )

            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "xr_state":
                        self.__handle_xr_state(message["data"])
                    elif message.get("type") == "console_log":
                        # Stream console logs from WebXR to terminal
                        log_data = message.get("data", {})
                        level = log_data.get("level", "log")
                        msg = log_data.get("message", "")

                        # Suppress spammy WS errors from WebXR
                        if "WS Error" in msg and "isTrusted" in msg:
                            continue

                        self.__logger.info(f"[WebXR:{level}] {msg}")
                    elif message.get("type") in {
                        "video_request",
                        "video_answer",
                        "video_ice",
                    }:
                        await self._handle_video_message(websocket, message)

            except WebSocketDisconnect:
                session = self.__video_sessions.pop(websocket, None)
                if session:
                    await session.close()
                self.__manager.disconnect(websocket)
                self.__logger.info("Client disconnected")

        self.__app.mount(mount_path, StaticFiles(directory=static_dir), name=mount_name)

    def run(self) -> None:
        """
        Runs the teleop server. This method is blocking.
        """
        self.__logger.info(self.__settings.model_dump_json())
        self.__logger.info(
            f"Server started at {self.__settings.host}:{self.__settings.port}"
        )
        self.__logger.info(
            f"The phone web app should be available at https://{get_local_ip()}:{self.__settings.port}"
        )

        ssl_keyfile = os.path.join(THIS_DIR, "key.pem")
        ssl_certfile = os.path.join(THIS_DIR, "cert.pem")

        uvicorn.run(
            self.__app,
            host=self.__settings.host,
            port=self.__settings.port,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            log_level="warning",
        )

    def stop(self) -> None:
        """
        Stops the teleop server.
        """
        # FastAPI/uvicorn handles shutdown automatically
        pass
