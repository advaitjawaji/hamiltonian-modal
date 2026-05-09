"""Modal encoder: projects joint-coordinate state to modal coordinates.

Computes η = Φ^T M (q - q_ref) and η̇ = Φ^T M q̇.
Mode shapes Φ are mass-orthonormal: Φ^T M Φ = I.
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import NewType

logger = logging.getLogger(__name__)

# Type aliases for documentation clarity
JointPos = NDArray[np.float64]    # shape (n_joints,)
JointVel = NDArray[np.float64]    # shape (n_joints,)
ModalCoord = NDArray[np.float64]  # shape (n_modes,)
ModalVel = NDArray[np.float64]    # shape (n_modes,)
MassMatrix = NDArray[np.float64]  # shape (n_joints, n_joints)
ModeShapes = NDArray[np.float64]  # shape (n_joints, n_modes)


def project_to_modal(
    q: JointPos,
    qdot: JointVel,
    mode_shapes: ModeShapes,
    mass_matrix: MassMatrix,
    q_ref: JointPos,
) -> tuple[ModalCoord, ModalVel]:
    """Project joint-coordinate state to modal coordinates.

    Computes η = Φ^T M (q - q_ref) and η̇ = Φ^T M q̇.

    Parameters
    ----------
    q : NDArray[np.float64], shape (n_joints,)
        Generalized joint positions.
    qdot : NDArray[np.float64], shape (n_joints,)
        Generalized joint velocities.
    mode_shapes : NDArray[np.float64], shape (n_joints, n_modes)
        Modal matrix Φ; columns are mass-orthonormal mode shapes.
    mass_matrix : NDArray[np.float64], shape (n_joints, n_joints)
        Mass matrix M(q_ref), symmetric positive definite.
    q_ref : NDArray[np.float64], shape (n_joints,)
        Reference configuration for the linearization.

    Returns
    -------
    eta : NDArray[np.float64], shape (n_modes,)
        Modal displacement coordinates.
    eta_dot : NDArray[np.float64], shape (n_modes,)
        Modal velocity coordinates.
    """
    q = np.asarray(q, dtype=np.float64)
    qdot = np.asarray(qdot, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    delta_q = q - q_ref
    eta = mode_shapes.T @ mass_matrix @ delta_q
    eta_dot = mode_shapes.T @ mass_matrix @ qdot
    return eta, eta_dot


def encode_position(
    q: JointPos,
    mode_shapes: ModeShapes,
    mass_matrix: MassMatrix,
    q_ref: JointPos,
) -> ModalCoord:
    """Encode joint positions to modal displacement coordinates."""
    delta_q = np.asarray(q, dtype=np.float64) - np.asarray(q_ref, dtype=np.float64)
    return np.asarray(mode_shapes, dtype=np.float64).T @ np.asarray(mass_matrix, dtype=np.float64) @ delta_q


def encode_velocity(
    qdot: JointVel,
    mode_shapes: ModeShapes,
    mass_matrix: MassMatrix,
) -> ModalVel:
    """Encode joint velocities to modal velocities."""
    return np.asarray(mode_shapes, dtype=np.float64).T @ np.asarray(mass_matrix, dtype=np.float64) @ np.asarray(qdot, dtype=np.float64)


def encode_momentum(
    qdot: JointVel,
    mode_shapes: ModeShapes,
) -> ModalVel:
    """Encode joint velocities to modal momenta p = Φ^T q̇ (mass-orthonormal)."""
    return np.asarray(mode_shapes, dtype=np.float64).T @ np.asarray(qdot, dtype=np.float64)


def batch_project_to_modal(
    q_batch: NDArray[np.float64],
    qdot_batch: NDArray[np.float64],
    mode_shapes: ModeShapes,
    mass_matrix: MassMatrix,
    q_ref: JointPos,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Batch version of project_to_modal.

    Parameters
    ----------
    q_batch : shape (batch, n_joints)
    qdot_batch : shape (batch, n_joints)
    mode_shapes : shape (n_joints, n_modes)
    mass_matrix : shape (n_joints, n_joints)
    q_ref : shape (n_joints,)

    Returns
    -------
    eta_batch : shape (batch, n_modes)
    eta_dot_batch : shape (batch, n_modes)
    """
    q_batch = np.asarray(q_batch, dtype=np.float64)
    qdot_batch = np.asarray(qdot_batch, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)

    delta_q = q_batch - q_ref[None, :]
    M_Phi = mass_matrix @ mode_shapes  # (n_joints, n_modes)
    eta_batch = delta_q @ M_Phi
    eta_dot_batch = qdot_batch @ M_Phi
    return eta_batch, eta_dot_batch
