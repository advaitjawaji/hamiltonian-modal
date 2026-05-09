"""Genesis integration interfaces.

Lightweight shim that lazy-loads Genesis so the rest of the package can be
imported and tested without a Genesis installation.  All public helpers raise
:class:`ImportError` with a descriptive message when Genesis is absent.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "load_genesis",
    "create_scene",
    "step_scene",
    "get_joint_positions",
    "get_joint_velocities",
    "apply_joint_torques",
]


def _require_genesis() -> Any:
    """Import and return the ``genesis`` module, raising on failure."""
    try:
        return import_module("genesis")
    except ImportError as exc:
        raise ImportError(
            "Genesis is required for simulation.  Install the project with "
            "robotics dependencies to use `hamiltonian_modal.utils.genesis_wrapper`."
        ) from exc


def load_genesis() -> Any:
    """Import Genesis and return the top-level module.

    Returns
    -------
    types.ModuleType
        The ``genesis`` module.

    Raises
    ------
    ImportError
        When Genesis is not installed.
    """
    return _require_genesis()


def create_scene(
    dt: float = 0.001,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    **kwargs: Any,
) -> Any:
    """Create a Genesis simulation scene.

    Parameters
    ----------
    dt:
        Simulation timestep in seconds.
    gravity:
        Gravity vector (x, y, z) in m s⁻².
    **kwargs:
        Additional keyword arguments forwarded to ``genesis.Scene``.

    Returns
    -------
    genesis.Scene
    """
    gs = _require_genesis()
    return gs.Scene(dt=dt, gravity=gravity, **kwargs)


def step_scene(scene: Any, n_steps: int = 1) -> None:
    """Advance the Genesis scene by *n_steps* simulation steps.

    Parameters
    ----------
    scene:
        A ``genesis.Scene`` instance.
    n_steps:
        Number of steps to advance.
    """
    for _ in range(n_steps):
        scene.step()


def get_joint_positions(robot: Any) -> NDArray[np.float64]:
    """Return the current joint positions from a Genesis robot entity.

    Parameters
    ----------
    robot:
        A Genesis robot / articulation object.

    Returns
    -------
    NDArray[np.float64]
        Joint positions, shape ``(n_dof,)``.
    """
    return np.asarray(robot.get_dofs_position(), dtype=np.float64)


def get_joint_velocities(robot: Any) -> NDArray[np.float64]:
    """Return the current joint velocities from a Genesis robot entity.

    Parameters
    ----------
    robot:
        A Genesis robot / articulation object.

    Returns
    -------
    NDArray[np.float64]
        Joint velocities, shape ``(n_dof,)``.
    """
    return np.asarray(robot.get_dofs_velocity(), dtype=np.float64)


def apply_joint_torques(robot: Any, torques: NDArray[np.float64]) -> None:
    """Apply generalised joint torques to a Genesis robot.

    Parameters
    ----------
    robot:
        A Genesis robot / articulation object.
    torques:
        Torque vector, shape ``(n_dof,)``.
    """
    robot.set_dofs_force(torques)
