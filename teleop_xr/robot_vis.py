import os
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .config import RobotVisConfig


class RobotVisModule:
    """
    Module for serving robot visualization assets (URDF, meshes) and broadcasting state.
    """

    def __init__(self, app: FastAPI, config: RobotVisConfig):
        self.app = app
        self.config = config
        self.logger = logging.getLogger("teleop.robot_vis")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/robot_assets/{file_path:path}")
        async def get_asset(file_path: str):
            self.logger.info(f"Asset request: {file_path}")
            full_path = ""

            if file_path == "robot.urdf":
                full_path = self.config.urdf_path
            elif "package://" in file_path:
                clean_path = file_path.split("package://")[-1]
                if self.config.mesh_path:
                    full_path = os.path.join(self.config.mesh_path, clean_path)
                else:
                    self.logger.warning(
                        f"Request for package resource '{file_path}' but 'mesh_path' is not configured."
                    )
                    full_path = clean_path
            else:
                # Try relative to URDF directory first
                urdf_dir = os.path.dirname(os.path.abspath(self.config.urdf_path))
                potential_paths = [os.path.join(urdf_dir, file_path)]

                # If mesh_path is configured, try resolving against it
                if self.config.mesh_path:
                    potential_paths.append(
                        os.path.join(self.config.mesh_path, file_path)
                    )

                full_path = potential_paths[0]
                for p in potential_paths:
                    if os.path.exists(p):
                        full_path = p
                        break

            if not os.path.exists(full_path):
                self.logger.warning(f"Asset not found: {full_path}")
                raise HTTPException(
                    status_code=404, detail=f"Asset not found: {file_path}"
                )

            media_type = None
            ext = os.path.splitext(full_path)[1].lower()
            if ext == ".stl":
                media_type = "application/octet-stream"
            elif ext == ".dae":
                media_type = "model/vnd.collada+xml"
            elif ext == ".obj":
                media_type = "text/plain"
            elif ext == ".urdf":
                media_type = "application/xml"

            return FileResponse(full_path, media_type=media_type)

    def get_frontend_config(self) -> Dict[str, Any]:
        return {
            "urdf_url": "/robot_assets/robot.urdf",
            "model_scale": self.config.model_scale,
            "initial_rotation_euler": self.config.initial_rotation_euler,
        }

    async def broadcast_state(self, connection_manager: Any, joints: Dict[str, float]):
        """
        Broadcasts the current joint state to all connected clients.

        Args:
            connection_manager: The ConnectionManager instance from Teleop class.
            joints: Dictionary mapping joint names to values (radians/meters).
        """
        message = {"type": "robot_state", "data": {"joints": joints}}
        await connection_manager.broadcast(json.dumps(message))
