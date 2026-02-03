import asyncio
import os
import math
import socket
import time
import logging
import json
from typing import Callable, List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import transforms3d as t3d
import numpy as np

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
from teleop_xr.robot_vis import RobotVisModule

TF_RUB2FLU = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"


def _resolve_frontend_paths(package_dir: str) -> tuple[str, str, str, str]:
    repo_root = os.path.abspath(os.path.join(package_dir, os.pardir))
    webxr_dist = os.path.join(repo_root, "webxr", "dist")
    webxr_index = os.path.join(webxr_dist, "index.html")

    # Prefer webxr/dist when running in development (webxr/ exists with dist/)
    if os.path.exists(webxr_index):
        return webxr_dist, webxr_index, "/", "webxr"

    # Fall back to package dist for installed package
    dist_dir = os.path.join(package_dir, "dist")
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
        self._lock = asyncio.Lock()

    async def register(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        async with self._lock:
            connections = list(self.active_connections)

        broken: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception:
                broken.append(connection)

        if broken:
            async with self._lock:
                for connection in broken:
                    if connection in self.active_connections:
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

        self.__relative_pose_init = None
        self.__absolute_pose_init = None
        self.__previous_received_pose = None
        self.__callbacks = []
        self.__pose = np.eye(4)

        euler = self.__settings.natural_phone_orientation_euler
        self.__natural_phone_pose = t3d.affines.compose(
            self.__settings.natural_phone_position,
            t3d.euler.euler2mat(euler[0], euler[1], euler[2]),
            [1, 1, 1],
        )

        self.__app = FastAPI()
        self.__app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.__manager = ConnectionManager()

        self.__ws_connect_lock = asyncio.Lock()
        self.__control_lock = asyncio.Lock()
        self.__controller_client_id: Optional[str] = None
        self.__controller_ws: Optional[WebSocket] = None
        self.__controller_last_seen_s: float = 0.0
        self.__ws_client_ids: dict[WebSocket, str] = {}

        self.__video_streams: list[VideoStreamConfig] = []
        self.__video_sources: dict[str, VideoSource] = video_sources or {}
        self.__video_sessions: dict[WebSocket, VideoStreamManager] = {}

        self.robot_vis: Optional[RobotVisModule] = None
        if self.__settings.robot_vis:
            self.robot_vis = RobotVisModule(self.__app, self.__settings.robot_vis)

        if self.__camera_views is not None and not self.__video_streams:
            self.set_video_streams(build_video_streams(self.__camera_views))

        # Configure logging
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        self.__setup_routes()

    @property
    def input_mode(self):
        return self.__settings.input_mode

    @property
    def app(self) -> FastAPI:
        return self.__app

    def set_pose(self, pose: np.ndarray) -> None:
        """
        Set the current pose of the end-effector.

        Parameters:
        - pose (np.ndarray): A 4x4 transformation matrix representing the pose.
        """
        self.__pose = pose

    def subscribe(self, callback: Callable[[np.ndarray, Any], None]) -> None:
        """
        Subscribe to receive updates from the teleop module.

        Parameters:
            callback (Callable[[np.ndarray, dict[str, Any]], None]): A callback function that will be called when pose updates are received.
                The callback function should take two arguments:
                    - np.ndarray: A 4x4 transformation matrix representing the end-effector target pose.
                    - dict[str, Any]: A dictionary containing additional information.
        """
        self.__callbacks.append(callback)

    def __notify_subscribers(self, pose, message):
        for callback in self.__callbacks:
            callback(pose, message)

    def set_video_streams(self, payload: Any) -> None:
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

    async def _handle_video_message(self, websocket: WebSocket, message: Any) -> None:
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

    def apply(self, position, orientation, move=True, scale=1.0, info=None):
        if info is None:
            info = {}

        position_arr = np.array([position["x"], position["y"], position["z"]])
        quat = np.array(
            [orientation["w"], orientation["x"], orientation["y"], orientation["z"]]
        )

        if not move:
            self.__relative_pose_init = None
            self.__absolute_pose_init = None
            self.__notify_subscribers(self.__pose, info)
            return

        received_pose_rub = t3d.affines.compose(
            position_arr, t3d.quaternions.quat2mat(quat), [1, 1, 1]
        )
        received_pose = TF_RUB2FLU @ received_pose_rub
        received_pose[:3, :3] = received_pose[:3, :3] @ np.linalg.inv(
            TF_RUB2FLU[:3, :3]
        )
        received_pose = received_pose @ self.__natural_phone_pose

        # Pose jump protection
        if self.__previous_received_pose is not None:
            if not are_close(
                received_pose,
                self.__previous_received_pose,
                lin_tol=0.05,
                ang_tol=math.radians(35),
            ):
                self.__logger.warning("Pose jump detected, resetting the pose")
                self.__relative_pose_init = None
                self.__previous_received_pose = received_pose
                return
        self.__previous_received_pose = received_pose

        # Accumulate the pose and publish
        if self.__relative_pose_init is None:
            self.__relative_pose_init = received_pose
            self.__absolute_pose_init = self.__pose
            self.__previous_received_pose = None

        assert self.__absolute_pose_init is not None
        relative_position = received_pose[:3, 3] - self.__relative_pose_init[:3, 3]
        relative_orientation = received_pose[:3, :3] @ np.linalg.inv(
            self.__relative_pose_init[:3, :3]
        )

        if scale > 1.0:
            relative_position *= scale

        self.__pose = np.eye(4)
        self.__pose[:3, 3] = self.__absolute_pose_init[:3, 3] + relative_position
        self.__pose[:3, :3] = relative_orientation @ self.__absolute_pose_init[:3, :3]

        # Apply scale
        if scale < 1.0:
            self.__pose = interpolate_transforms(
                self.__absolute_pose_init, self.__pose, scale
            )

        # Notify the subscribers
        self.__notify_subscribers(self.__pose, info)

    def __handle_xr_state(self, message):
        input_mode = self.__settings.input_mode
        devices = message.get("devices", [])

        # Log fetch latency if present
        fetch_latency = message.get("fetch_latency_ms")
        if fetch_latency is not None:
            self.__logger.debug(f"XR input fetch latency: {fetch_latency:.2f}ms")

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

    async def publish_joint_state(self, joints: Dict[str, float]):
        if self.robot_vis:
            await self.robot_vis.broadcast_state(self.__manager, joints)

    def __setup_routes(self):
        static_dir, index_path, mount_path, mount_name = _resolve_frontend_paths(
            THIS_DIR
        )

        @self.__app.get("/")
        async def index():
            return FileResponse(index_path)

        @self.__app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            control_timeout_s = 5.0

            await websocket.accept()

            try:
                await asyncio.wait_for(self.__ws_connect_lock.acquire(), timeout=0.05)
            except asyncio.TimeoutError:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "connection_error",
                            "data": {
                                "reason": "connecting",
                                "message": "WebSocket busy: another client is connecting.",
                            },
                        }
                    )
                )
                await websocket.close()
                return

            try:
                await self.__manager.register(websocket)

                # Send config message on connect
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "config",
                            "data": self.__settings.model_dump(),
                        }
                    )
                )

                if self.robot_vis:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "robot_config",
                                "data": self.robot_vis.get_frontend_config(),
                            }
                        )
                    )

                if self.__video_streams:
                    await self.__manager.send_personal_message(
                        json.dumps(
                            {
                                "type": "video_config",
                                "data": {
                                    "streams": [
                                        s.__dict__ for s in self.__video_streams
                                    ]
                                },
                            }
                        ),
                        websocket,
                    )
            finally:
                self.__ws_connect_lock.release()

            self.__logger.info("Client connected")

            async def close_video_sessions_for_client_id(
                client_id_to_close: str,
            ) -> None:
                to_close: list[tuple[WebSocket, VideoStreamManager]] = []
                for ws, session in list(self.__video_sessions.items()):
                    ws_id = self.__ws_client_ids.get(ws)
                    if ws_id == client_id_to_close:
                        to_close.append((ws, session))

                for ws, session in to_close:
                    await session.close()
                    self.__video_sessions.pop(ws, None)

            async def close_video_sessions_not_matching(controller_id: str) -> None:
                to_close: list[tuple[WebSocket, VideoStreamManager]] = []
                for ws, session in list(self.__video_sessions.items()):
                    ws_id = self.__ws_client_ids.get(ws)
                    if ws_id != controller_id:
                        to_close.append((ws, session))

                for ws, session in to_close:
                    await session.close()
                    self.__video_sessions.pop(ws, None)

            async def check_or_claim_control(
                claimed_client_id: str,
            ) -> tuple[bool, Optional[str]]:
                expired_controller: Optional[str] = None
                newly_claimed: Optional[str] = None

                async with self.__control_lock:
                    now = time.monotonic()
                    if (
                        self.__controller_client_id is not None
                        and (now - self.__controller_last_seen_s) > control_timeout_s
                    ):
                        expired_controller = self.__controller_client_id
                        self.__controller_client_id = None
                        self.__controller_ws = None
                        self.__controller_last_seen_s = 0.0

                    if self.__controller_client_id is None:
                        self.__controller_client_id = claimed_client_id
                        newly_claimed = claimed_client_id

                    in_control = self.__controller_client_id == claimed_client_id
                    if in_control:
                        self.__controller_ws = websocket
                        self.__controller_last_seen_s = now

                    controller_id = self.__controller_client_id

                if expired_controller is not None:
                    await close_video_sessions_for_client_id(expired_controller)

                if newly_claimed is not None:
                    await close_video_sessions_not_matching(newly_claimed)

                return in_control, controller_id

            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    msg_type = message.get("type")
                    client_id = message.get("client_id")
                    if isinstance(client_id, str) and client_id:
                        self.__ws_client_ids[websocket] = client_id

                    if msg_type == "control_check":
                        if not isinstance(client_id, str) or not client_id:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "control_status",
                                        "data": {
                                            "in_control": False,
                                            "controller_client_id": self.__controller_client_id,
                                        },
                                    }
                                )
                            )
                            continue

                        in_control, controller_id = await check_or_claim_control(
                            client_id
                        )

                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "control_status",
                                    "data": {
                                        "in_control": in_control,
                                        "controller_client_id": controller_id,
                                    },
                                }
                            )
                        )
                        continue

                    if msg_type == "xr_state":
                        if not isinstance(client_id, str) or not client_id:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "deny",
                                        "data": {
                                            "reason": "missing_client_id",
                                            "controller_client_id": self.__controller_client_id,
                                        },
                                    }
                                )
                            )
                            continue

                        allowed, controller_id = await check_or_claim_control(client_id)
                        if not allowed:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "deny",
                                        "data": {
                                            "reason": "not_in_control",
                                            "controller_client_id": controller_id,
                                        },
                                    }
                                )
                            )
                            continue

                        self.__handle_xr_state(message["data"])
                        continue

                    if msg_type in {"video_request", "video_answer", "video_ice"}:
                        if not isinstance(client_id, str) or not client_id:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "deny",
                                        "data": {
                                            "reason": "missing_client_id",
                                            "controller_client_id": self.__controller_client_id,
                                        },
                                    }
                                )
                            )
                            continue

                        allowed, controller_id = await check_or_claim_control(client_id)
                        if not allowed:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "deny",
                                        "data": {
                                            "reason": "not_in_control",
                                            "controller_client_id": controller_id,
                                        },
                                    }
                                )
                            )
                            continue

                        await self._handle_video_message(websocket, message)
                        continue

                    if msg_type == "console_log":
                        # Stream console logs from WebXR to terminal
                        log_data = message.get("data", {})
                        level = log_data.get("level", "log")
                        msg = log_data.get("message", "")

                        # Suppress spammy WS errors from WebXR
                        if "WS Error" in msg and "isTrusted" in msg:
                            continue

                        self.__logger.info(f"[WebXR:{level}] {msg}")

            except WebSocketDisconnect:
                pass
            finally:
                disconnected_client_id = self.__ws_client_ids.pop(websocket, None)
                if disconnected_client_id is not None:
                    async with self.__control_lock:
                        if (
                            self.__controller_ws is websocket
                            and self.__controller_client_id == disconnected_client_id
                        ):
                            self.__controller_client_id = None
                            self.__controller_ws = None
                            self.__controller_last_seen_s = 0.0

                session = self.__video_sessions.pop(websocket, None)
                if session:
                    await session.close()
                await self.__manager.disconnect(websocket)
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
