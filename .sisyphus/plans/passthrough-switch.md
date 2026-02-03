# Passthrough switch in-session

## Goal
- Allow toggling passthrough to switch between AR/VR while an XR session is active.

## Plan
- Review current XR session setup and passthrough toggle flow.
- Update XR session handling to restart session when passthrough changes in-session.
- Keep UI switch active during XR sessions and wire it to the session restart.
- Validate with targeted frontend tests and capture UI evidence.
