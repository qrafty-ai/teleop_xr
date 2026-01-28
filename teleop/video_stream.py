from dataclasses import dataclass
from typing import Any
import asyncio
import cv2
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from aiortc.mediastreams import VideoStreamTrack


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


class CameraStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, config: VideoStreamConfig):
        super().__init__()
        self.stream_id = config.id
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def _ensure_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            self._capture = cv2.VideoCapture(self._config.device)
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        return self._capture

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        capture = self._ensure_capture()
        ok, frame = capture.read()
        if not ok:
            await asyncio.sleep(0.01)
            return await self.recv()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


def build_tracks(configs: list[VideoStreamConfig]) -> list[CameraStreamTrack]:
    return [CameraStreamTrack(cfg) for cfg in configs if cfg.enabled]


class VideoStreamManager:
    def __init__(
        self, configs: list[VideoStreamConfig], ice_servers: list[str] | None = None
    ):
        servers = [RTCIceServer(urls=url) for url in (ice_servers or [])]
        self._pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        self._configs = configs

    async def create_offer(self) -> RTCSessionDescription:
        for track in build_tracks(self._configs):
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


def route_video_message(manager: VideoStreamManager, message: dict) -> Any:
    msg_type = message.get("type")
    data = message.get("data", {})
    if msg_type == "video_answer":
        return manager.handle_answer(data.get("sdp", ""), data.get("type", "answer"))
    if msg_type == "video_ice":
        return manager.add_ice(data)
