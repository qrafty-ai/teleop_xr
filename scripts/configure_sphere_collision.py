import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tyro
import viser
import yourdfpy
from ballpark import (
    BallparkConfig,
    RefineParams,
    Robot,
    RobotSpheresResult,
    Sphere,
    SpherePreset,
    SpherizeParams,
    SPHERE_COLORS,
)
from loguru import logger
from viser.extras import ViserUrdf

from teleop_xr.ik.loader import load_robot_class


def _process_chunk_worker(
    robot: Robot,
    count: int,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
    non_contiguous: list[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Worker function for parallel collision check."""
    local_counts = {pair: 0 for pair in non_contiguous}
    # Re-seed random generator in worker process
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

    for _ in range(count):
        cfg = np.random.uniform(lower, upper)
        distances = robot.get_mesh_distances(cfg)
        for pair, dist in distances.items():
            if dist <= threshold:
                local_counts[pair] += 1
    return local_counts


def compute_collision_ignore_pairs(
    robot: Robot,
    n_samples: int = 1000,
    collision_threshold: float = 0.0,
    n_jobs: int = 1,
) -> list[list[str]]:
    """Compute collision ignore pairs by sampling random configurations.

    Samples random joint configurations and checks mesh-mesh distances
    between non-contiguous link pairs using the ORIGINAL collision meshes.
    Pairs that are ALWAYS in collision or NEVER in collision across all
    samples should be ignored (they provide no useful signal for the
    collision checker).

    Args:
        robot: Ballpark Robot instance with loaded collision meshes.
        n_samples: Number of random configurations to sample.
        collision_threshold: Distance threshold below which a pair is
            considered "in collision". 0.0 means actual mesh intersection.
        n_jobs: Number of parallel jobs (processes) to use.

    Returns:
        List of [link_a, link_b] pairs that should be ignored.
    """
    lower, upper = robot.joint_limits
    non_contiguous = robot.non_contiguous_pairs

    if not non_contiguous:
        logger.info("No non-contiguous pairs to check.")
        return []

    collision_counts: dict[tuple[str, str], int] = {pair: 0 for pair in non_contiguous}

    logger.info(
        f"Sampling {n_samples} configurations to compute collision ignore pairs "
        f"({len(non_contiguous)} non-contiguous pairs)..."
    )

    if n_jobs > 1:
        chunk_size = (n_samples + n_jobs - 1) // n_jobs
        chunks = [min(chunk_size, n_samples - i * chunk_size) for i in range(n_jobs)]
        chunks = [c for c in chunks if c > 0]

        logger.info(f"Parallel execution with {n_jobs} jobs (chunks: {chunks})")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _process_chunk_worker,
                    robot,
                    c,
                    lower,
                    upper,
                    collision_threshold,
                    non_contiguous,
                )
                for c in chunks
            ]
            results = [f.result() for f in futures]

        for res in results:
            for pair, count in res.items():
                collision_counts[pair] += count
    else:
        for i in range(n_samples):
            cfg = np.random.uniform(lower, upper)
            distances = robot.get_mesh_distances(cfg)

            for pair, dist in distances.items():
                if dist <= collision_threshold:
                    collision_counts[pair] += 1

            if (i + 1) % 100 == 0:
                logger.debug(f"  Sampled {i + 1}/{n_samples} configurations")

    always_colliding: list[list[str]] = []
    never_colliding: list[list[str]] = []

    for pair, count in collision_counts.items():
        if count == n_samples:
            always_colliding.append(list(pair))
        elif count == 0:
            never_colliding.append(list(pair))

    ignore_pairs = always_colliding + never_colliding

    logger.info(
        f"Collision ignore pair analysis complete:\n"
        f"  Always colliding: {len(always_colliding)} pairs\n"
        f"  Never colliding:  {len(never_colliding)} pairs\n"
        f"  Active (keep):    {len(non_contiguous) - len(ignore_pairs)} pairs\n"
        f"  Total ignored:    {len(ignore_pairs)} pairs"
    )

    return ignore_pairs


class _SpheresGui:
    """GUI controls for sphere visualization."""

    def __init__(
        self, server: viser.ViserServer, robot: Robot, default_export_path: str
    ):
        self._server = server
        self._robot = robot
        self._export_callback: Callable[[], None] | None = None

        # Track state for change detection
        self._last_mode: str = "Auto"
        self._last_total: int = 40
        self._last_link_budgets: dict[str, int] = {}
        self._last_show: bool = True
        self._last_opacity: float = 0.9
        self._last_refine: bool = False
        self._last_preset: str = "Balanced"
        self._last_params: dict[str, float] = {}
        self._last_joint_config: np.ndarray | None = None  # Track joint config changes
        self._needs_spherize = True  # Allocation or spherize params changed
        self._needs_refine_update = True  # Refine toggle or refine params changed
        self._needs_visual_update = True

        # Collision pair tracking
        self._mesh_distances: dict[tuple[str, str], float] = {}
        self._skipped_pairs: set[tuple[str, str]] = set()  # User-confirmed skips

        # New state for collision tab
        self._generated_ignore_pairs: list[list[str]] | None = None
        self._highlight_handles: list[viser.SceneNodeHandle] = []
        self._last_highlight_link: str = "None"
        self._last_show_highlights: bool = True
        self._user_fully_disabled_links: set[str] = set()
        self._disabled_list_handles: list[Any] = []

        # Current config (updated by presets or custom sliders)
        self._current_config = BallparkConfig.from_preset(SpherePreset.BALANCED)

        # Params folder handle and sliders (created dynamically for Custom mode)
        self._params_folder: viser.GuiFolderHandle | None = None
        self._params_sliders: dict[str, viser.GuiInputHandle[Any]] = {}
        self._config_folder: viser.GuiFolderHandle | None = None

        # Build GUI
        tab_group = server.gui.add_tab_group()

        # Spheres tab
        with tab_group.add_tab("Spheres"):
            with server.gui.add_folder("Visualization"):
                self._show_spheres = server.gui.add_checkbox(
                    "Show Spheres", initial_value=True
                )
                self._opacity = server.gui.add_slider(
                    "Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.9
                )
                self._refine = server.gui.add_checkbox(
                    "Refine (optimize)", initial_value=False
                )

            # Config folder - preset and parameters
            self._config_folder = server.gui.add_folder("Config")
            with self._config_folder:
                self._preset = server.gui.add_dropdown(
                    "Preset",
                    options=["Balanced", "Conservative", "Surface", "Custom"],
                    initial_value="Balanced",
                )

            with server.gui.add_folder("Allocation"):
                self._mode = server.gui.add_dropdown(
                    "Mode", options=["Auto", "Manual"], initial_value="Auto"
                )
                self._total_spheres = server.gui.add_slider(
                    "Target #", min=0, max=256, step=1, initial_value=64
                )
                self._sphere_count_number = server.gui.add_number(
                    "Actual #", initial_value=0, disabled=True
                )
                self._link_sliders: dict[str, viser.GuiInputHandle[Any]] = {}
                with server.gui.add_folder("Per-Link", expand_by_default=False):
                    for link_name in robot.collision_links:
                        display = (
                            link_name[:20] + "..." if len(link_name) > 20 else link_name
                        )
                        self._link_sliders[link_name] = server.gui.add_slider(
                            display,
                            min=0,
                            max=50,
                            step=1,
                            initial_value=1,
                            disabled=True,
                        )

            with server.gui.add_folder("Export"):
                self._export_filename = server.gui.add_text(
                    "Filename", initial_value=default_export_path
                )
                export_button = server.gui.add_button("Export to JSON")

                @export_button.on_click
                def _(_) -> None:
                    if self._export_callback:
                        self._export_callback()

        # Collision Tab
        with tab_group.add_tab("Collision"):
            with server.gui.add_folder("Settings"):
                self._n_samples_slider = server.gui.add_slider(
                    "Samples", min=100, max=5000, step=100, initial_value=1000
                )
                self._threshold_slider = server.gui.add_slider(
                    "Threshold", min=0.0, max=0.1, step=0.001, initial_value=0.01
                )
                self._threads_slider = server.gui.add_slider(
                    "Threads", min=1, max=32, step=1, initial_value=10
                )

            with server.gui.add_folder("Calculation"):
                self._calc_button = server.gui.add_button("Calculate Ignore Pairs")
                self._calc_status = server.gui.add_text(
                    "Status", initial_value="Not calculated", disabled=True
                )

                @self._calc_button.on_click
                def _(_) -> None:
                    self._calc_status.value = "Calculating..."
                    try:
                        pairs = compute_collision_ignore_pairs(
                            self._robot,
                            n_samples=self._n_samples_slider.value,
                            collision_threshold=self._threshold_slider.value,
                            n_jobs=self._threads_slider.value,
                        )
                        self._generated_ignore_pairs = pairs
                        self._calc_status.value = f"Calculated ({len(pairs)} pairs)"
                        self._needs_visual_update = True
                    except Exception as e:
                        self._calc_status.value = f"Error: {e}"
                        logger.error(f"Calculation failed: {e}")

            with server.gui.add_folder("Manual Overrides"):
                self._manual_disable_dropdown = server.gui.add_dropdown(
                    "Link to Disable", options=sorted(robot.collision_links)
                )
                self._manual_disable_button = server.gui.add_button(
                    "Disable All Collision"
                )
                self._manual_disabled_list_folder = server.gui.add_folder(
                    "Disabled List"
                )

                @self._manual_disable_button.on_click
                def _(_) -> None:
                    link = self._manual_disable_dropdown.value
                    if link not in self._user_fully_disabled_links:
                        self._user_fully_disabled_links.add(link)
                        self._update_disabled_list_ui()
                        self._update_highlights()

            with server.gui.add_folder("Visualization"):
                self._show_highlights = server.gui.add_checkbox(
                    "Show Highlights", initial_value=True
                )
                self._highlight_link_dropdown = server.gui.add_dropdown(
                    "Select Link",
                    options=["None"] + sorted(robot.collision_links),
                    initial_value="None",
                )
                self._ignored_links_text = server.gui.add_text(
                    "Ignored Links", initial_value="-", disabled=True
                )

        # Joints tab
        lower, upper = robot.joint_limits
        self._joint_sliders = []
        with tab_group.add_tab("Joints"):
            for i in range(len(lower)):
                slider = server.gui.add_slider(
                    f"Joint {i}",
                    min=float(lower[i]),
                    max=float(upper[i]),
                    step=0.01,
                    initial_value=(float(lower[i]) + float(upper[i])) / 2,
                )
                self._joint_sliders.append(slider)

    def poll(self) -> None:
        """Check for GUI changes and update internal state."""
        # Preset change - affects both spherize and refine params
        if self._preset.value != self._last_preset:
            self._last_preset = self._preset.value
            self._apply_preset(self._preset.value)
            self._needs_spherize = True
            self._needs_refine_update = True

        # Custom params change (only in Custom mode)
        if self._preset.value == "Custom" and self._params_sliders:
            current_params = self._get_params_values()
            if current_params != self._last_params:
                # Check which params changed
                spherize_params = {
                    "padding",
                    "target_tightness",
                    "aspect_threshold",
                    "percentile",
                    "max_radius_ratio",
                    "uniform_radius",
                    "axis_mode",
                    "symmetry_mode",
                    "symmetry_tolerance",
                }
                refine_params = {
                    "n_iters",
                    "tol",
                    "lambda_under",
                    "lambda_over",
                    "lambda_center_reg",
                    "lambda_radius_reg",
                    "lambda_self_collision",
                }

                for key in current_params:
                    if current_params[key] != self._last_params.get(key):
                        if key in spherize_params:
                            self._needs_spherize = True
                        if key in refine_params:
                            self._needs_refine_update = True

                self._last_params = current_params

        # Mode change - affects allocation
        if self._mode.value != self._last_mode:
            self._last_mode = self._mode.value
            is_manual = self._last_mode == "Manual"
            for slider in self._link_sliders.values():
                slider.disabled = not is_manual
            self._total_spheres.disabled = is_manual
            self._needs_spherize = True

        # Total spheres change (auto mode)
        if self._mode.value == "Auto" and self._total_spheres.value != self._last_total:
            self._last_total = int(self._total_spheres.value)
            self._needs_spherize = True

        # Per-link slider change (manual mode)
        if self._mode.value == "Manual":
            current = {name: int(s.value) for name, s in self._link_sliders.items()}
            if current != self._last_link_budgets:
                # Sync similar links: when one link in a similarity group changes,
                # update all others in the group to match
                for name, new_val in current.items():
                    if new_val != self._last_link_budgets.get(name, 0):
                        group = self._get_group_for_link(name)
                        if group and len(group) > 1:
                            for other in group:
                                if other != name and other in self._link_sliders:
                                    self._link_sliders[other].value = new_val
                current = {name: int(s.value) for name, s in self._link_sliders.items()}
                self._total_spheres.value = sum(current.values())
                self._last_link_budgets = current
                self._needs_spherize = True

        # Refine checkbox change - ONLY affects refine, not spherize
        if self._refine.value != self._last_refine:
            self._last_refine = self._refine.value
            self._needs_refine_update = True  # NOT _needs_spherize

        # Visibility/opacity change
        if (
            self._show_spheres.value != self._last_show
            or self._opacity.value != self._last_opacity
        ):
            self._last_show = self._show_spheres.value
            self._last_opacity = self._opacity.value
            self._needs_visual_update = True

        # Joint config change - triggers re-refine (mesh distances depend on config)
        current_joint_config = self.joint_config
        config_changed = False
        if self._last_joint_config is None or not np.allclose(
            current_joint_config, self._last_joint_config, atol=1e-4
        ):
            self._last_joint_config = current_joint_config.copy()
            self._needs_refine_update = True
            config_changed = True

        # Update highlights
        current_highlight = self._highlight_link_dropdown.value
        current_show = self._show_highlights.value

        if (
            self._needs_visual_update
            or config_changed
            or current_highlight != self._last_highlight_link
            or current_show != self._last_show_highlights
        ):
            self._last_highlight_link = current_highlight
            self._last_show_highlights = current_show
            self._update_highlights()

    def _get_group_for_link(self, link_name: str) -> list[str] | None:
        for group in self._robot._similarity.groups:
            if link_name in group:
                return group
        return None

    def _create_params_folder(self) -> None:
        """Create the Params folder with all sliders for Custom mode."""
        if self._params_folder is not None:
            return  # Already exists
        if self._config_folder is None:
            return  # Config folder not initialized

        cfg = self._current_config

        with self._config_folder:
            self._params_folder = self._server.gui.add_folder("Params")

        with self._params_folder:
            # Spherize parameters
            with self._server.gui.add_folder("Spherize"):
                self._params_sliders["padding"] = self._server.gui.add_slider(
                    "padding",
                    min=1.0,
                    max=1.2,
                    step=0.01,
                    initial_value=cfg.spherize.padding,
                )
                self._params_sliders["target_tightness"] = self._server.gui.add_slider(
                    "target_tightness",
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    initial_value=cfg.spherize.target_tightness,
                )
                self._params_sliders["aspect_threshold"] = self._server.gui.add_slider(
                    "aspect_threshold",
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    initial_value=cfg.spherize.aspect_threshold,
                )
                self._params_sliders["percentile"] = self._server.gui.add_slider(
                    "percentile",
                    min=90.0,
                    max=100.0,
                    step=0.5,
                    initial_value=cfg.spherize.percentile,
                )
                self._params_sliders["max_radius_ratio"] = self._server.gui.add_slider(
                    "max_radius_ratio",
                    min=0.2,
                    max=0.8,
                    step=0.05,
                    initial_value=cfg.spherize.max_radius_ratio,
                )
                self._params_sliders["uniform_radius"] = self._server.gui.add_checkbox(
                    "uniform_radius",
                    initial_value=cfg.spherize.uniform_radius,
                )
                self._params_sliders["axis_mode"] = self._server.gui.add_dropdown(
                    "axis_mode",
                    options=["aligned", "pca"],
                    initial_value=cfg.spherize.axis_mode,
                )
                self._params_sliders["symmetry_mode"] = self._server.gui.add_dropdown(
                    "symmetry_mode",
                    options=["auto", "off", "force"],
                    initial_value=cfg.spherize.symmetry_mode,
                )
                self._params_sliders["symmetry_tolerance"] = (
                    self._server.gui.add_slider(
                        "symmetry_tolerance",
                        min=0.01,
                        max=0.2,
                        step=0.01,
                        initial_value=cfg.spherize.symmetry_tolerance,
                    )
                )

            # Refine optimization parameters
            with self._server.gui.add_folder("Optimization"):
                self._params_sliders["n_iters"] = self._server.gui.add_slider(
                    "n_iters",
                    min=10,
                    max=500,
                    step=10,
                    initial_value=cfg.refine.n_iters,
                )
                self._params_sliders["tol"] = self._server.gui.add_slider(
                    "tol",
                    min=1e-6,
                    max=1e-2,
                    step=1e-5,
                    initial_value=cfg.refine.tol,
                )

            # Loss weights
            with self._server.gui.add_folder("Losses"):
                self._params_sliders["lambda_under"] = self._server.gui.add_slider(
                    "lambda_under",
                    min=0.0,
                    max=5.0,
                    step=0.001,
                    initial_value=cfg.refine.lambda_under,
                )
                self._params_sliders["lambda_over"] = self._server.gui.add_slider(
                    "lambda_over",
                    min=0.0,
                    max=0.1,
                    step=0.001,
                    initial_value=cfg.refine.lambda_over,
                )
                self._params_sliders["lambda_center_reg"] = self._server.gui.add_slider(
                    "lambda_center_reg",
                    min=0.0,
                    max=10.0,
                    step=0.001,
                    initial_value=cfg.refine.lambda_center_reg,
                )
                self._params_sliders["lambda_radius_reg"] = self._server.gui.add_slider(
                    "lambda_radius_reg",
                    min=0.0,
                    max=10.0,
                    step=0.001,
                    initial_value=cfg.refine.lambda_radius_reg,
                )
                self._params_sliders["lambda_self_collision"] = (
                    self._server.gui.add_slider(
                        "lambda_self_collision",
                        min=0.0,
                        max=10.0,
                        step=0.001,
                        initial_value=cfg.refine.lambda_self_collision,
                    )
                )

        # Cache initial values
        self._last_params = self._get_params_values()

    def _remove_params_folder(self) -> None:
        """Remove the Params folder."""
        if self._params_folder is not None:
            self._params_folder.remove()
            self._params_folder = None
            self._params_sliders.clear()
            self._last_params.clear()

    def _get_params_values(self) -> dict[str, float]:
        """Get current parameter values from sliders."""
        if not self._params_sliders:
            return {}
        return {name: s.value for name, s in self._params_sliders.items()}

    def _apply_preset(self, preset_name: str) -> None:
        """Apply a configuration preset."""
        if preset_name == "Custom":
            # Custom mode: show Params folder
            self._create_params_folder()
        else:
            # Preset mode: hide Params folder and load config
            self._remove_params_folder()
            preset_map = {
                "Balanced": SpherePreset.BALANCED,
                "Conservative": SpherePreset.CONSERVATIVE,
                "Surface": SpherePreset.SURFACE,
            }
            self._current_config = BallparkConfig.from_preset(preset_map[preset_name])

    def get_config(self) -> BallparkConfig:
        """Get the current configuration (from preset or custom sliders)."""
        if self._preset.value != "Custom":
            return self._current_config

        # Build config from slider values
        p = self._params_sliders
        return BallparkConfig(
            spherize=SpherizeParams(
                padding=float(p["padding"].value),
                target_tightness=float(p["target_tightness"].value),
                aspect_threshold=float(p["aspect_threshold"].value),
                percentile=float(p["percentile"].value),
                max_radius_ratio=float(p["max_radius_ratio"].value),
                uniform_radius=bool(p["uniform_radius"].value),
                axis_mode=str(p["axis_mode"].value),
                symmetry_mode=str(p["symmetry_mode"].value),
                symmetry_tolerance=float(p["symmetry_tolerance"].value),
            ),
            refine=RefineParams(
                n_iters=int(p["n_iters"].value),
                tol=float(p["tol"].value),
                lambda_under=float(p["lambda_under"].value),
                lambda_over=float(p["lambda_over"].value),
                lambda_center_reg=float(p["lambda_center_reg"].value),
                lambda_radius_reg=float(p["lambda_radius_reg"].value),
                lambda_self_collision=float(p["lambda_self_collision"].value),
            ),
        )

    @property
    def is_auto_mode(self) -> bool:
        return self._mode.value == "Auto"

    @property
    def total_spheres(self) -> int:
        return int(self._total_spheres.value)

    @property
    def manual_allocation(self) -> dict[str, int]:
        """Per-link allocation from manual sliders."""
        return {name: int(s.value) for name, s in self._link_sliders.items()}

    def update_sliders_from_allocation(self, alloc: dict[str, int]) -> None:
        """Update per-link sliders to reflect an allocation."""
        for name, slider in self._link_sliders.items():
            slider.value = alloc.get(name, 0)
        self._last_link_budgets = alloc

    def update_mesh_distances(self, distances: dict[tuple[str, str], float]) -> None:
        """Update the collision pair info with mesh distances."""
        self._mesh_distances = distances

    @property
    def excluded_collision_pairs(self) -> set[tuple[str, str]]:
        """Return user-skipped pairs."""
        return self._skipped_pairs.copy()

    @property
    def needs_spherize(self) -> bool:
        return self._needs_spherize

    def mark_spherized(self) -> None:
        self._needs_spherize = False

    @property
    def needs_refine_update(self) -> bool:
        return self._needs_refine_update

    def set_needs_refine_update(self) -> None:
        self._needs_refine_update = True

    def mark_refine_updated(self) -> None:
        self._needs_refine_update = False

    @property
    def needs_visual_update(self) -> bool:
        return self._needs_visual_update

    def mark_visuals_updated(self) -> None:
        self._needs_visual_update = False

    @property
    def show_spheres(self) -> bool:
        return self._show_spheres.value

    @property
    def opacity(self) -> float:
        return self._opacity.value

    @property
    def refine_enabled(self) -> bool:
        return self._refine.value

    @property
    def joint_config(self) -> np.ndarray:
        return np.array([s.value for s in self._joint_sliders])

    @property
    def export_filename(self) -> str:
        return self._export_filename.value

    def on_export(self, callback: Callable[[], None]) -> None:
        self._export_callback = callback

    def _update_highlights(self) -> None:
        """Update visualization of highlighted/ignored links."""
        # Clear existing
        for handle in self._highlight_handles:
            handle.remove()
        self._highlight_handles.clear()

        if (
            not self._show_highlights.value
            or self._highlight_link_dropdown.value == "None"
        ):
            self._ignored_links_text.value = "-"
            return

        selected_link = self._highlight_link_dropdown.value

        # Identify ignored links
        ignored_links = set()

        # 1. Computed ignore pairs
        if self._generated_ignore_pairs:
            for pair in self._generated_ignore_pairs:
                if selected_link in pair:
                    other = pair[0] if pair[1] == selected_link else pair[1]
                    ignored_links.add(other)

        # 2. Adjacent links (always ignored)
        try:
            # Accessing protected member to get adjacency info
            adj_pairs = self._robot._get_adjacent_links()
            for pair in adj_pairs:
                if selected_link in pair:
                    other = pair[0] if pair[1] == selected_link else pair[1]
                    ignored_links.add(other)
        except AttributeError:
            pass

        # 3. Manually disabled links
        for disabled in self._user_fully_disabled_links:
            if selected_link == disabled:
                for other in self._robot.collision_links:
                    if other != selected_link:
                        ignored_links.add(other)
            else:
                ignored_links.add(disabled)

        self._ignored_links_text.value = (
            ", ".join(sorted(ignored_links)) if ignored_links else "None"
        )

        # Get current transforms
        cfg = self.joint_config
        transforms = self._robot.compute_transforms(cfg)
        link_name_to_idx = {name: i for i, name in enumerate(self._robot.links)}

        # Helper to add mesh
        def add_highlight(link_name: str, color: tuple[float, float, float]) -> None:
            if link_name not in link_name_to_idx:
                return
            idx = link_name_to_idx[link_name]
            T = transforms[idx]

            # Get mesh from ballpark robot
            mesh = self._robot._link_meshes.get(link_name)
            if mesh is None:
                return

            handle = self._server.scene.add_mesh_simple(
                f"/highlights/{link_name}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=color,
                opacity=0.5,
                position=T[4:],
                wxyz=T[:4],
            )
            self._highlight_handles.append(handle)

        # Highlight selected (Green)
        add_highlight(selected_link, (0.0, 1.0, 0.0))

        # Highlight ignored (Red)
        for link in ignored_links:
            add_highlight(link, (1.0, 0.0, 0.0))

    def update_sphere_count(self, actual: int) -> None:
        """Update the sphere count display."""
        self._sphere_count_number.value = actual

    def _update_disabled_list_ui(self) -> None:
        # Clear existing
        for handle in self._disabled_list_handles:
            handle.remove()
        self._disabled_list_handles.clear()

        with self._manual_disabled_list_folder:
            if not self._user_fully_disabled_links:
                h = self._server.gui.add_text(
                    "Status", initial_value="No manually disabled links.", disabled=True
                )
                self._disabled_list_handles.append(h)
                return

            for link in sorted(self._user_fully_disabled_links):
                btn = self._server.gui.add_button(f"Remove {link}", color="red")
                self._disabled_list_handles.append(btn)

                def make_cb(link_name: str) -> Callable[[Any], None]:
                    return lambda _: self._remove_disabled_link(link_name)

                btn.on_click(make_cb(link))

    def _remove_disabled_link(self, link: str) -> None:
        if link in self._user_fully_disabled_links:
            self._user_fully_disabled_links.remove(link)
            self._update_disabled_list_ui()
            self._update_highlights()


class _SphereVisuals:
    """Manages sphere visualization in viser."""

    def __init__(self, server: viser.ViserServer, link_names: list[str]):
        self._server = server
        self._link_names = link_names
        self._frames: dict[str, viser.FrameHandle] = {}
        self._handles: dict[str, viser.IcosphereHandle] = {}
        self._link_spheres: dict[str, list[Sphere]] = {}

    def update(
        self,
        result: RobotSpheresResult,
        opacity: float,
        visible: bool,
    ) -> None:
        """Rebuild sphere visuals from result."""
        # Clear existing
        for h in self._handles.values():
            h.remove()
        for f in self._frames.values():
            f.remove()
        self._handles.clear()
        self._frames.clear()
        self._link_spheres = result.link_spheres

        if not visible:
            return

        for link_idx, link_name in enumerate(self._link_names):
            spheres = self._link_spheres.get(link_name, [])
            if not spheres:
                continue

            color = SPHERE_COLORS[link_idx % len(SPHERE_COLORS)]
            rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            for sphere_idx, sphere in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"
                frame = self._server.scene.add_frame(
                    f"/sphere_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                self._frames[key] = frame
                center = sphere.center
                self._handles[key] = self._server.scene.add_icosphere(
                    f"/sphere_frames/{key}/sphere",
                    radius=float(sphere.radius),
                    position=(float(center[0]), float(center[1]), float(center[2])),
                    color=rgb,
                    opacity=opacity,
                )

    def update_transforms(self, Ts_link_world: np.ndarray) -> None:
        """Update sphere positions from link transforms."""
        for link_idx, link_name in enumerate(self._link_names):
            spheres = self._link_spheres.get(link_name, [])
            if not spheres:
                continue

            T = Ts_link_world[link_idx]
            wxyz, pos = T[:4], T[4:]

            for sphere_idx in range(len(spheres)):
                key = f"{link_name}_{sphere_idx}"
                if key in self._frames:
                    self._frames[key].wxyz = wxyz
                    self._frames[key].position = pos


def _export_collision_data(
    gui: _SpheresGui,
    robot: Robot,
    result: RobotSpheresResult | None,
) -> None:
    """Export collision data to JSON."""
    if not result:
        return

    path = Path(gui.export_filename)

    spheres_data = {}
    for link_name, spheres in result.link_spheres.items():
        spheres_data[link_name] = {
            "centers": [s.center.tolist() for s in spheres],
            "radii": [float(s.radius) for s in spheres],
        }

    logger.info("Computing collision ignore pairs (this may take a moment)...")
    if gui._generated_ignore_pairs is not None:
        ignore_pairs = gui._generated_ignore_pairs
    else:
        ignore_pairs = compute_collision_ignore_pairs(
            robot,
            n_samples=gui._n_samples_slider.value,
            collision_threshold=gui._threshold_slider.value,
            n_jobs=gui._threads_slider.value,
        )

    # Add manually disabled pairs
    all_links = set(robot.collision_links)
    manual_pairs = set()
    for disabled in gui._user_fully_disabled_links:
        for other in all_links:
            if disabled == other:
                continue
            pair = tuple(sorted((disabled, other)))
            manual_pairs.add(pair)

    # Merge with computed pairs
    final_pairs = set(tuple(sorted(p)) for p in ignore_pairs)
    final_pairs.update(manual_pairs)

    export_pairs = [list(p) for p in sorted(final_pairs)]

    collision_data = {
        "spheres": spheres_data,
        "collision_ignore_pairs": export_pairs,
    }

    logger.info(f"Exporting to {path}")
    with open(path, "w") as f:
        json.dump(collision_data, f, indent=2)
    logger.success(
        f"Exported {result.num_spheres} spheres and {len(ignore_pairs)} ignore pairs"
    )


def _run_loop_step(
    gui: _SpheresGui,
    robot: Robot,
    sphere_visuals: _SphereVisuals,
    urdf_vis: ViserUrdf,
    result: RobotSpheresResult | None,
) -> RobotSpheresResult | None:
    """Run one iteration of the main loop."""
    gui.poll()

    if gui.needs_spherize:
        if gui.is_auto_mode:
            if gui.total_spheres == 0:
                allocation = {name: 0 for name in robot.collision_links}
            else:
                allocation = robot.auto_allocate(gui.total_spheres)
            gui.update_sliders_from_allocation(allocation)
        else:
            allocation = gui.manual_allocation

        config = gui.get_config()
        if gui.total_spheres > 0:
            t0 = time.perf_counter()
            result = robot.spherize(allocation=allocation, config=config)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Generated {result.num_spheres} spheres in {elapsed:.1f}ms")
        else:
            result = RobotSpheresResult(link_spheres={})

        gui.update_sphere_count(result.num_spheres)
        gui.mark_spherized()
        gui.set_needs_refine_update()

    if gui.needs_refine_update and result is not None:
        config = gui.get_config()
        if gui.refine_enabled and result.num_spheres > 0:
            t0 = time.perf_counter()
            refined = robot.refine(
                result,
                config=config,
                joint_cfg=gui.joint_config,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Refined spheres in {elapsed:.1f}ms")
            sphere_visuals.update(refined, gui.opacity, gui.show_spheres)
        else:
            sphere_visuals.update(result, gui.opacity, gui.show_spheres)

        gui.mark_refine_updated()
        gui.mark_visuals_updated()

    if gui.needs_visual_update and result:
        sphere_visuals.update(result, gui.opacity, gui.show_spheres)
        gui.mark_visuals_updated()

    cfg = gui.joint_config
    urdf_vis.update_cfg(cfg)

    if gui.show_spheres:
        Ts = robot.compute_transforms(cfg)
        sphere_visuals.update_transforms(Ts)

    return result


def main(
    robot_class: str = "h1",
    target_spheres: int = 64,
) -> None:
    """
    Generate sphere decomposition for a robot with interactive GUI.

    Args:
        robot_class: The robot class specification (e.g., 'h1', 'UnitreeH1Robot' or 'module:Class').
        target_spheres: Initial target number of spheres for decomposition.
    """
    logger.info(f"Loading robot class: {robot_class}")
    try:
        RobotCls = load_robot_class(robot_class)
        robot_inst = RobotCls()
    except Exception as e:
        logger.error(f"Failed to load robot: {e}")
        return

    robot_name = getattr(robot_inst, "name", robot_class)
    vis_config = robot_inst.get_vis_config()
    if vis_config is None:
        logger.error(
            f"Robot {robot_name} does not provide a visualization config (urdf_path missing)."
        )
        return

    urdf_path = vis_config.urdf_path
    logger.info(f"Generating spheres for {robot_name} using URDF: {urdf_path}")

    # Load URDF with collision meshes
    urdf_coll = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

    # Create ballpark Robot instance
    ballpark_robot = Robot(urdf_coll)

    # Determine default save path
    import teleop_xr.ik.robot as robot_mod

    current_dir = os.path.dirname(os.path.abspath(robot_mod.__file__))
    asset_dir = os.path.join(current_dir, "robots", "assets", robot_name)
    os.makedirs(asset_dir, exist_ok=True)
    default_export_path = os.path.join(asset_dir, "collision.json")

    # Set up viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf_coll, root_node_name="/robot")

    gui = _SpheresGui(server, ballpark_robot, default_export_path)
    sphere_visuals = _SphereVisuals(server, ballpark_robot.links)

    # Initial target spheres
    gui._total_spheres.value = target_spheres

    # Current sphere result
    result: RobotSpheresResult | None = None

    def on_export() -> None:
        # Capture current result
        _export_collision_data(gui, ballpark_robot, result)

    gui.on_export(on_export)

    logger.info("Starting visualization (open browser to view)...")
    while True:
        result = _run_loop_step(gui, ballpark_robot, sphere_visuals, urdf_vis, result)
        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
