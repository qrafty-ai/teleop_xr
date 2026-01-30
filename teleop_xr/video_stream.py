from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import asyncio
import threading
import time

import cv2
import numpy as np
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
    if not isinstance(streams, list):
        raise ValueError("streams must be a list")
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


@runtime_checkable
class VideoSource(Protocol):
    new_frame_event: threading.Event

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class OpenCVVideoSource:
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

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

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

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.read_lock:
            frame = (
                self.frame.copy() if self.grabbed and self.frame is not None else None
            )
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self) -> None:
        self.started = False
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.cap.release()


class ExternalVideoSource:
    def __init__(self):
        self.frame: np.ndarray | None = None
        self.grabbed = False
        self.read_lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.read_lock:
            frame = (
                self.frame.copy() if self.grabbed and self.frame is not None else None
            )
            grabbed = self.grabbed
        return grabbed, frame

    def put_frame(self, frame: np.ndarray) -> None:
        with self.read_lock:
            self.frame = frame
            self.grabbed = True
        self.new_frame_event.set()


class CameraStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, source: VideoSource, stream_id: str):
        super().__init__()
        self.source = source
        self.stream_id = stream_id

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        # Ensure source is started
        self.source.start()

        # Wait for new frame
        while not self.source.new_frame_event.is_set():
            await asyncio.sleep(0.001)

        self.source.new_frame_event.clear()
        ok, frame = self.source.read()

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
        self.source.stop()


def build_sources(configs: list[VideoStreamConfig]) -> dict[str, VideoSource]:
    sources = {}
    for cfg in configs:
        if cfg.enabled:
            sources[cfg.id] = OpenCVVideoSource(
                cfg.device,
                cfg.width,
                cfg.height,
                cfg.fps,
            )
    return sources


class VideoStreamManager:
    def __init__(
        self, sources: dict[str, VideoSource], ice_servers: list[str] | None = None
    ):
        servers = [RTCIceServer(urls=url) for url in (ice_servers or [])]
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        self._sources = sources
        self._tracks: list[CameraStreamTrack] = []

    async def create_offer(self) -> RTCSessionDescription:
        self._tracks = [
            CameraStreamTrack(source, stream_id)
            for stream_id, source in self._sources.items()
        ]
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
