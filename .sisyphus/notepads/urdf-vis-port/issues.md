## Route Collision with Vite
- Date: 2026-01-31
- Problem: Robot visualization assets used the `/assets/` route, which collided with Vite's default build output directory (`/assets/`). This caused requests for frontend chunks to be intercepted by `robot_vis.py`, leading to 'Asset not found' warnings and potentially breaking the frontend.
