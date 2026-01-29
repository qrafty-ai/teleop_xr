# TeleopXR

**TeleopXR** is a powerful WebXR-based interface for robot teleoperation, transforming your VR/AR headset or phone into a precise robot controller.

## Overview

This project provides a seamless bridge between WebXR-capable devices (Quest 3, Vision Pro, modern smartphones) and robotic arms. It features a lightweight, high-performance web client that communicates with a Python backend to control robots like xArm, UR5e, and Lite6.

## Key Improvements

This release represents a polished and streamlined version of the original teleop system:

- **Production-Ready Client**: Removed extraneous test objects (plants, robots, banners) for a clean, professional AR/VR environment.
- **Optimized Performance**: Streamlined asset loading and scene management.
- **Focus on Control**: Dedicated UI for camera views and robot status monitoring.

## Installation

### Prerequisites
- Python 3.8+
- Node.js 18+ (for client development)
- A WebXR-compatible device

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/teleop-xr.git
   cd teleop-xr
   ```

2. Install the Python package:
   ```bash
   pip install -e .
   ```

3. (Optional) Build the WebXR client:
   ```bash
   cd webxr
   npm install
   npm run build
   ```

## Usage

Start the basic teleoperation server:

```bash
python -m teleop_xr.basic
```

Then open the WebXR client in your headset browser (usually served at `https://localhost:8000` or configured URL).

## Acknowledgements

This project is a fork of [teleop](https://github.com/SpesRobotics/teleop) by SpesRobotics. We gratefully acknowledge their work in pioneering WebXR-based robot teleoperation.

## License

[License Information]
