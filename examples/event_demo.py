"""
Example showing how to use the TeleopXR event system to handle button interactions.
"""

import sys
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.events import EventProcessor, EventSettings, ButtonEvent


def main():
    # 1. Configure Teleop
    # By default, this uses host="0.0.0.0" and port=4433
    settings = TeleopSettings()
    teleop = Teleop(settings=settings)

    # 2. Configure Event System
    event_settings = EventSettings(
        double_press_threshold_ms=300,
        long_press_threshold_ms=500,
    )
    processor = EventProcessor(event_settings)

    # 3. Register Callbacks

    def on_any_button_down(event: ButtonEvent):
        print(f"[EVENT] {event.controller.value} {event.button.value} DOWN")

    def on_any_button_up(event: ButtonEvent):
        print(
            f"[EVENT] {event.controller.value} {event.button.value} UP (held {event.hold_duration_ms:.0f}ms)"
        )

    def on_generic_double_press(event: ButtonEvent):
        print(f"*** DOUBLE PRESS: {event.controller.value} {event.button.value} ***")

    def on_generic_long_press(event: ButtonEvent):
        print(f"*** LONG PRESS: {event.controller.value} {event.button.value} ***")

    # Catch-all callbacks for any button down/up
    processor.on_button_down(callback=on_any_button_down)
    processor.on_button_up(callback=on_any_button_up)

    # Generic callbacks for double/long press detection on ALL buttons
    processor.on_double_press(callback=on_generic_double_press)
    processor.on_long_press(callback=on_generic_long_press)

    # 4. Subscribe and Run
    print("Event system demo starting...")
    print("1. Open the WebXR app in your headset.")
    print("2. Enter VR mode.")
    print("3. Try pressing, double-pressing, and holding buttons.")
    print("Press Ctrl+C to exit.\n")

    teleop.subscribe(processor.process)

    try:
        teleop.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
