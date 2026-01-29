from dataclasses import dataclass
from .config import InputMode


@dataclass
class CommonCLI:
    host: str = "0.0.0.0"
    port: int = 4443
    input_mode: InputMode = InputMode.CONTROLLER
