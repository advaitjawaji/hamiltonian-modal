"""Modal decoder: projects modal coordinates back to joint space.

Computes q = q_ref + Φ η and q̇ = Φ η̇.
"""
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def project_from_modal(
    eta: NDArray[np.float64],
    eta_dot: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    q_ref: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project modal coordinates back to joint-coordinate state.

    Computes q = q_ref + Φ η and q̇ = Φ η̇.

    Parameters
    ----------
    eta : NDArray[np.float64], shape (n_modes,)
        Modal displacement coordinates.
    eta_dot : NDArray[np.float64], shape (n_modes,)
        Modal velocities.
    mode_shapes : NDArray[np.float64], shape (n_joints, n_modes)
        Modal matrix Φ.
    q_ref : NDArray[np.float64], shape (n_joints,)
        Reference configuration.

    Returns
    -------
    q : NDArray[np.float64], shape (n_joints,)
        Reconstructed joint positions.
    qdot : NDArray[np.float64], shape (n_joints,)
        Reconstructed joint velocities.
    """
    eta = np.asarray(eta, dtype=np.float64)
    eta_dot = np.asarray(eta_dot, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    q = q_ref + mode_shapes @ eta
    qdot = mode_shapes @ eta_dot
    return q, qdot


def decode_position(
    eta: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    q_ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode modal displacement to joint position."""
    return np.asarray(q_ref, dtype=np.float64) + np.asarray(mode_shapes, dtype=np.float64) @ np.asarray(eta, dtype=np.float64)


def decode_velocity(
    eta_dot: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode modal velocity to joint velocity."""
    return np.asarray(mode_shapes, dtype=np.float64) @ np.asarray(eta_dot, dtype=np.float64)


def decode_momentum(
    mu: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode modal momentum to joint velocity (mass-orthonormal)."""
    return np.asarray(mode_shapes, dtype=np.float64) @ np.asarray(mu, dtype=np.float64)


def batch_project_from_modal(
    eta_batch: NDArray[np.float64],
    eta_dot_batch: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    q_ref: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Batch version of project_from_modal.

    Parameters
    ----------
    eta_batch : shape (batch, n_modes)
    eta_dot_batch : shape (batch, n_modes)
    mode_shapes : shape (n_joints, n_modes)
    q_ref : shape (n_joints,)

    Returns
    -------
    q_batch : shape (batch, n_joints)
    qdot_batch : shape (batch, n_joints)
    """
    eta_batch = np.asarray(eta_batch, dtype=np.float64)
    eta_dot_batch = np.asarray(eta_dot_batch, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    q_batch = q_ref[None, :] + eta_batch @ mode_shapes.T
    qdot_batch = eta_dot_batch @ mode_shapes.T
    return q_batch, qdot_batch
