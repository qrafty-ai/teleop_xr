"""
Utilities for optional IK dependencies.
"""

import sys


def ensure_ik_dependencies():
    """
    Check and configure JAX for IK mode.

    Raises:
        SystemExit: If JAX is not installed, exits with code 1.
    """
    try:
        import jax

        jax.config.update("jax_platform_name", "cpu")
    except ImportError:
        print(
            "Error: JAX is required for IK mode. "
            "Install with: pip install 'teleop-xr[ik]'",
            file=sys.stderr,
        )
        sys.exit(1)


def list_robots_or_exit():
    """
    List available robots and exit.

    Raises:
        SystemExit: Always exits after listing robots (or on error).
    """
    from loguru import logger

    try:
        robots = list_available_robots()
        logger.info("Available robots (via entry points):")
        if not robots:
            logger.info("  None")
        for name, path in robots.items():
            logger.info(f"  {name}: {path}")
    except ImportError:
        logger.error(
            "IK dependencies not installed. Install with: pip install 'teleop-xr[ik]'"
        )
        sys.exit(1)
    sys.exit(0)


def list_available_robots():
    """Return the robots entry points exposed by the IK module."""
    try:
        from teleop_xr.ik.loader import list_available_robots as _load_robots

        return _load_robots()
    except ImportError:
        raise
