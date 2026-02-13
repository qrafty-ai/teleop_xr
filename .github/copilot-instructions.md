# Copilot Instructions

## Project overview

- Python backend lives in `teleop_xr/`.
- WebXR frontend lives in `webxr/` (Next.js + TypeScript).

## Setup

- Python dependencies: `uv sync`
- WebXR dependencies: `cd webxr && npm install`

## Build & test

- Python tests: `python -m pytest`
- WebXR lint: `cd webxr && npm run lint`
- WebXR tests: `cd webxr && npm run test`
- WebXR build: `cd webxr && npm run build`

## Repository notes

- Keep changes minimal and focused.
- Follow existing patterns and file layout.
- If you touch `.sisyphus/` plan or notepad files,
  commit them together with your changes.
