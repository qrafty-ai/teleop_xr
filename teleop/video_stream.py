from dataclasses import dataclass
from typing import Any
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame


@dataclass(frozen=True)
class VideoStreamConfig:
    id: str
    device: int | str
    width: int = 1280
    height: int = 720
    fps: int = 30
    codec: str = "vp8"
    bitrate_kbps: int = 1500
    enabled: bool = True


def parse_video_config(payload: dict[str, Any]) -> list[VideoStreamConfig]:
    streams = payload.get("streams", [])
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    ids: set[str] = set()
    configs: list[VideoStreamConfig] = []
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("stream entries must be objects")
        stream_id = stream.get("id")
        if not stream_id or stream_id in ids:
            raise ValueError("stream id missing or duplicate")
        ids.add(stream_id)
        device = stream.get("device", 0)
        configs.append(
            VideoStreamConfig(
                id=str(stream_id),
                device=device,
                width=int(stream.get("width", 1280)),
                height=int(stream.get("height", 720)),
                fps=int(stream.get("fps", 30)),
                codec=str(stream.get("codec", "vp8")),
                bitrate_kbps=int(stream.get("bitrate_kbps", 1500)),
                enabled=bool(stream.get("enabled", True)),
            )
        )
    return configs


class ThreadedVideoCapture:
    def __init__(self, src: int | str, width: int, height: int, fps: int):
        self.src = src
        # Use V4L2 backend for Linux performance
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # Minimize buffering
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Determine format (MJPG is faster for high res, YUYV/default otherwise)
        # For now, stick to default or try MJPG if supported
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.read_lock:
                    self.frame = frame
                    self.grabbed = True
                self.new_frame_event.set()
            else:
                # If read fails, wait briefly to avoid busy loop
                time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.grabbed else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.cap.release()


class CameraStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, config: VideoStreamConfig):
        super().__init__()
        self.stream_id = config.id
        self._config = config
        self._capture: ThreadedVideoCapture | None = None

    def _ensure_capture(self) -> ThreadedVideoCapture:
        if self._capture is None:
            self._capture = ThreadedVideoCapture(
                self._config.device,
                self._config.width,
                self._config.height,
                self._config.fps,
            )
            self._capture.start()
        return self._capture

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        capture = self._ensure_capture()

        # Wait for new frame
        while not capture.new_frame_event.is_set():
            await asyncio.sleep(0.001)

        capture.new_frame_event.clear()
        ok, frame = capture.read()

        if not ok or frame is None:
            # If failed to get frame, wait and retry
            await asyncio.sleep(0.01)
            return await self.recv()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        if self._capture:
            self._capture.stop()
            self._capture = None


def build_tracks(configs: list[VideoStreamConfig]) -> list[CameraStreamTrack]:
    return [CameraStreamTrack(cfg) for cfg in configs if cfg.enabled]


class VideoStreamManager:
    def __init__(
        self, configs: list[VideoStreamConfig], ice_servers: list[str] | None = None
    ):
        servers = [RTCIceServer(urls=url) for url in (ice_servers or [])]
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        self._configs = configs
        self._tracks: list[CameraStreamTrack] = []

    async def create_offer(self) -> RTCSessionDescription:
        self._tracks = build_tracks(self._configs)
        for track in self._tracks:
            self._pc.addTrack(track)
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        return self._pc.localDescription

    async def handle_answer(self, sdp: str, sdp_type: str = "answer") -> None:
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=sdp_type)
        )

    async def add_ice(self, candidate: dict) -> None:
        if candidate and candidate.get("candidate"):
            ice = candidate_from_sdp(candidate["candidate"])
            ice.sdpMid = candidate.get("sdpMid")
            ice.sdpMLineIndex = candidate.get("sdpMLineIndex")
            await self._pc.addIceCandidate(ice)

    async def close(self) -> None:
        await self._pc.close()
        for track in self._tracks:
            track.stop()
        self._tracks = []


def route_video_message(manager: VideoStreamManager, message: dict) -> Any:
    msg_type = message.get("type")
    data = message.get("data", {})
    if msg_type == "video_answer":
        return manager.handle_answer(data.get("sdp", ""), data.get("type", "answer"))
    if msg_type == "video_ice":
        return manager.add_ice(data)
