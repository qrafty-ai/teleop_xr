from teleop_xr.common_cli import CommonCLI
from teleop_xr.config import InputMode


def test_common_cli_defaults():
    cli = CommonCLI()
    assert cli.host == "0.0.0.0"
    assert cli.port == 4443
    assert cli.input_mode == InputMode.CONTROLLER


def test_common_cli_init():
    cli = CommonCLI(host="127.0.0.1", port=8000, input_mode=InputMode.HAND)
    assert cli.host == "127.0.0.1"
    assert cli.port == 8000
    assert cli.input_mode == InputMode.HAND
