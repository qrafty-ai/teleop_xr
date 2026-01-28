from dataclasses import dataclass
from typing import Any


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
