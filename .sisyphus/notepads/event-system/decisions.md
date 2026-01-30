# Decisions - Event System

## ButtonDetector Implementation
- Added `ButtonEventType` enum to distinguish between different types of button events (`BUTTON_DOWN`, `BUTTON_UP`, `DOUBLE_PRESS`, `LONG_PRESS`).
- `ButtonDetector.update` returns a list of `ButtonEvent` objects, allowing multiple events to be triggered by a single state change (e.g., `BUTTON_DOWN` and `DOUBLE_PRESS`).
- Implemented double-press debouncing by resetting `last_release_time_ms` to 0 after a double-press is detected and released. This ensures that a sequence like Press-Release-Press-Release-Press (all within threshold) only triggers one double-press for the first pair.
- Long-press events are fired exactly once when the threshold is reached, and not repeated while the button is held.
- State is tracked independently per `(button, controller)` pair using a dictionary.
