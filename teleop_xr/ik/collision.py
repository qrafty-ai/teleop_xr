import hashlib
import json
import numpy as np
import trimesh
import yourdfpy
import ballpark
from ballpark import SpherizeParams
from loguru import logger
import io


def validate_sphere_decomposition(decomposition: dict) -> bool:
    """
    Validates that the decomposition dictionary matches the expected schema:
    { "link_name": { "centers": [[x, y, z], ...], "radii": [r, ...] } }
    """
    if not isinstance(decomposition, dict):
        return False
    for link_name, data in decomposition.items():
        if not isinstance(data, dict):
            raise ValueError(f"Invalid data for link {link_name}")
        if "centers" not in data:
            raise ValueError(f"Missing centers for link {link_name}")
        if "radii" not in data:
            raise ValueError(f"Missing radii for link {link_name}")
        centers = data["centers"]
        radii = data["radii"]
        if not isinstance(centers, list) or not isinstance(radii, list):
            raise ValueError(f"Centers and radii must be lists for link {link_name}")
        if len(centers) != len(radii):
            raise ValueError(
                f"Length mismatch between centers and radii for link {link_name}"
            )
    return True


def get_decomposition_cache_key(decomposition: dict) -> str:
    """
    Generates a deterministic hash key for a sphere decomposition.
    """
    sorted_links = sorted(decomposition.keys())
    stable_data = []
    for link in sorted_links:
        link_data = decomposition[link]
        # Round to 6 decimal places for float stability
        centers = np.array(link_data["centers"]).round(6).tolist()
        radii = np.array(link_data["radii"]).round(6).tolist()
        stable_data.append((link, centers, radii))

    # JSON dump with sort_keys=True for stability
    s = json.dumps(stable_data, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def _get_primitive_mesh(geometry):
    if geometry.box is not None:
        return trimesh.creation.box(extents=geometry.box.size)
    if geometry.cylinder is not None:
        return trimesh.creation.cylinder(
            radius=geometry.cylinder.radius, height=geometry.cylinder.length
        )
    if geometry.sphere is not None:
        return trimesh.creation.uv_sphere(radius=geometry.sphere.radius)
    return None


def generate_collision_spheres(
    urdf_path=None, urdf_string=None, n_spheres_per_link=8, padding=0.0
):
    """
    Generates sphere-based collision decomposition for a robot URDF.

    Args:
        urdf_path: Path to the URDF file.
        urdf_string: URDF content as a string (overrides urdf_path).
        n_spheres_per_link: Number of spheres to decompose each link into.
        padding: Additive padding for sphere radii.

    Returns:
        A dictionary with the sphere decomposition schema.
    """
    # Set seed for reproducibility where possible
    np.random.seed(42)

    if urdf_string is not None:
        urdf = yourdfpy.URDF.load(io.StringIO(urdf_string))
    elif urdf_path is not None:
        urdf = yourdfpy.URDF.load(urdf_path)
    else:
        raise ValueError("Either urdf_path or urdf_string must be provided.")

    decomposition = {}

    # ballpark params for strict over-approximation
    # Use a small padding multiplier and high sample count to ensure strict coverage
    params = SpherizeParams(percentile=100.0, padding=1.05, n_samples=10000)

    for link in urdf.robot.links:
        # Special case: single sphere primitive for accuracy and efficiency
        if len(link.collisions) == 1 and link.collisions[0].geometry.sphere is not None:
            coll = link.collisions[0]
            origin = coll.origin if coll.origin is not None else np.eye(4)
            radius = coll.geometry.sphere.radius + padding
            center = origin[:3, 3]
            decomposition[link.name] = {
                "centers": [center.tolist()],
                "radii": [radius],
            }
            continue

        link_meshes = []

        for collision in link.collisions:
            origin = collision.origin if collision.origin is not None else np.eye(4)
            geom = collision.geometry

            mesh = None
            if geom.mesh is not None:
                try:
                    resolved_path = urdf._filename_handler(geom.mesh.filename)
                    mesh = trimesh.load(resolved_path)
                    if geom.mesh.scale is not None:
                        mesh.apply_scale(geom.mesh.scale)
                except Exception as e:
                    logger.error(
                        f"Failed to load mesh {geom.mesh.filename} for link {link.name}: {e}"
                    )
            else:
                mesh = _get_primitive_mesh(geom)

            if mesh is not None:
                # Apply origin transform (link frame to collision frame)
                mesh.apply_transform(origin)
                link_meshes.append(mesh)

        if link_meshes:
            combined_mesh = trimesh.util.concatenate(link_meshes)

            try:
                spheres = ballpark.spherize(
                    combined_mesh, n_spheres_per_link, params=params
                )
                # Convert from jax arrays to numpy
                centers = np.array([np.array(s.center) for s in spheres])
                radii = np.array([float(s.radius) for s in spheres])

                # Apply additive padding
                radii += padding

                # Apply a small extra margin for strict coverage if needed
                # (ballpark uses samples, so it might miss some vertices)
                radii *= 1.05

                # Strict coverage check
                vertices = combined_mesh.vertices
                if len(vertices) > 0:
                    # Sample some vertices if there are too many for memory-efficient check
                    if len(vertices) > 10000:
                        sample_idx = np.random.choice(
                            len(vertices), 10000, replace=False
                        )
                        test_verts = vertices[sample_idx]
                    else:
                        test_verts = vertices

                    dists = np.linalg.norm(
                        test_verts[:, None, :] - centers[None, :, :], axis=2
                    )
                    covered = np.any(dists <= (radii[None, :] + 1e-3), axis=1)

                    if not np.all(covered):
                        logger.warning(
                            f"Strict coverage check failed for link {link.name}, falling back to bounding sphere"
                        )
                        center = combined_mesh.bounding_sphere.centroid
                        radius = (
                            combined_mesh.bounding_sphere.primitive.radius + padding
                        )
                        centers = np.array([center])
                        radii = np.array([radius])

            except Exception as e:
                logger.error(
                    f"Ballpark failed for link {link.name}: {e}. Falling back to bounding sphere."
                )
                center = combined_mesh.bounding_sphere.centroid
                radius = combined_mesh.bounding_sphere.primitive.radius + padding
                centers = np.array([center])
                radii = np.array([radius])

            # Deterministic sorting of spheres within the link
            # Sort by x, then y, then z
            idx = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))

            decomposition[link.name] = {
                "centers": centers[idx].tolist(),
                "radii": radii[idx].tolist(),
            }

    return decomposition
