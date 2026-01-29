import logging


def parse_device_spec(value: str | int) -> int | str:
    """
    Accept int -> return int
    Accept string digits -> return int
    Accept string starting with '/dev/' -> return string
    Reject empty string, None, bool, or other values with ValueError
    """
    if value is None or isinstance(value, bool):
        raise ValueError("Device spec cannot be None or bool")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Device spec cannot be empty string")
        if value.isdigit():
            return int(value)
        if value.startswith("/dev/"):
            return value

    raise ValueError(f"Invalid device spec: {value}")


def build_camera_views_config(head=None, wrist_left=None, wrist_right=None) -> dict:
    """
    Inputs are optional device specs (int/str/None)
    Returns dict with keys for provided values only: 'head', 'wrist_left', 'wrist_right'
    Each entry is { "device": <normalized> }
    If the same device is mapped to multiple views, allow it but emit a logging.warning
    """
    inputs = {"head": head, "wrist_left": wrist_left, "wrist_right": wrist_right}

    config = {}
    device_to_views = {}

    for view_key, raw_value in inputs.items():
        if raw_value is not None:
            normalized = parse_device_spec(raw_value)
            config[view_key] = {"device": normalized}

            if normalized not in device_to_views:
                device_to_views[normalized] = []
            device_to_views[normalized].append(view_key)

    for device, views in device_to_views.items():
        if len(views) > 1:
            logging.warning(
                f"Device {device} is mapped to multiple views: {', '.join(views)}"
            )

    return config


def build_video_streams(camera_views: dict) -> dict:
    """
    Returns { "streams": [{"id": <view_key>, "device": <device>}, ...] }
    Preserve stable order: head, wrist_left, wrist_right if present
    """
    order = ["head", "wrist_left", "wrist_right"]
    streams = []

    for view_key in order:
        if view_key in camera_views:
            streams.append({"id": view_key, "device": camera_views[view_key]["device"]})

    return {"streams": streams}
