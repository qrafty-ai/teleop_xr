import mujoco
import urdf2mjcf.convert
import tempfile
import os


def load_model(path: str) -> mujoco.MjModel:
    """
    Loads a robot model from a URDF or MJCF (.xml) file.

    Args:
        path: Path to the model file.

    Returns:
        A mujoco.MjModel instance.
    """
    if path.endswith(".urdf"):
        # Convert URDF to MJCF at runtime
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "model.xml")
            urdf2mjcf.convert.convert_urdf_to_mjcf(path, tmp_path, copy_meshes=True)
            return mujoco.MjModel.from_xml_path(tmp_path)
    elif path.endswith(".xml"):
        # Load MJCF directly
        return mujoco.MjModel.from_xml_path(path)
    else:
        raise ValueError(
            f"Unsupported file format: {path}. Only .urdf and .xml are supported."
        )
