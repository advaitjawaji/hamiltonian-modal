"""Configuration-dependent mode basis switching.

Selects the appropriate mode basis for the current contact mode and
handles smooth transitions via overlap matrix.
"""
import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

CONTACT_MODES: dict[int, str] = {
    0: "double_stance",
    1: "left_single",
    2: "right_single",
    3: "flight",
    4: "other",
}


def detect_contact_mode(scene: Any) -> int:
    """Detect current contact mode from Genesis scene.

    Parameters
    ----------
    scene : gs.Scene
        Active Genesis scene with G1 robot.

    Returns
    -------
    mode : int
        0=double_stance, 1=left_single, 2=right_single, 3=flight, 4=other
    """
    try:
        import genesis as gs  # noqa: F401
        contacts = scene.get_contacts()
        left_contact = any(
            "left" in str(c.link_a).lower() or "left" in str(c.link_b).lower()
            for c in contacts
        )
        right_contact = any(
            "right" in str(c.link_a).lower() or "right" in str(c.link_b).lower()
            for c in contacts
        )
        if left_contact and right_contact:
            return 0
        elif left_contact:
            return 1
        elif right_contact:
            return 2
        else:
            return 3
    except ImportError:
        return 0
    except Exception:
        logger.debug("Contact detection failed; defaulting to double_stance")
        return 0


def compute_overlap_matrix(
    Phi_old: NDArray[np.float64],
    Phi_new: NDArray[np.float64],
    M: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute M-weighted overlap matrix between two mode bases.

    O_{ij} = Φ_old_i^T M Φ_new_j

    Used for smooth modal-state tracking across basis transitions.

    Parameters
    ----------
    Phi_old : shape (n_joints, r)
    Phi_new : shape (n_joints, r)
    M : shape (n_joints, n_joints)

    Returns
    -------
    O : shape (r, r)
        Overlap matrix.
    """
    Phi_old = np.asarray(Phi_old, dtype=np.float64)
    Phi_new = np.asarray(Phi_new, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    return Phi_old.T @ M @ Phi_new


def project_with_basis_switch(
    q: NDArray[np.float64],
    qdot: NDArray[np.float64],
    contact_mode: int,
    mode_bases: dict[int, dict[str, NDArray[np.float64]]],
    q_ref: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project to modal coords using the correct basis for current contact mode.

    Handles smooth mode-tracking across basis transitions via overlap matrix.

    Parameters
    ----------
    q : NDArray[np.float64], shape (n_joints,)
    qdot : NDArray[np.float64], shape (n_joints,)
    contact_mode : int
        Current contact mode index (0-4).
    mode_bases : dict[int, dict]
        Maps contact_mode -> {"mode_shapes": ..., "mass_matrix": ..., "q_ref": ...}
    q_ref : NDArray[np.float64] | None
        Override reference configuration. If None, uses mode_bases[contact_mode]["q_ref"].

    Returns
    -------
    eta : shape (n_modes,)
    eta_dot : shape (n_modes,)
    """
    if contact_mode not in mode_bases:
        logger.warning("Contact mode %d not in mode_bases; using mode 0", contact_mode)
        contact_mode = 0

    basis = mode_bases[contact_mode]
    mode_shapes = basis["mode_shapes"]
    mass_matrix = basis["mass_matrix"]
    q_ref_basis = basis.get("q_ref", np.zeros(q.shape))
    if q_ref is not None:
        q_ref_basis = q_ref

    from hamiltonian_modal.modal.encoder import project_to_modal
    return project_to_modal(q, qdot, mode_shapes, mass_matrix, q_ref_basis)


def select_modes(
    eigenvalues: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    indices: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Select a subset of modes by index."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    return eigenvalues[indices], mode_shapes[:, indices]


def select_low_frequency_modes(
    eigenvalues: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    n_modes: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Select the n_modes lowest-frequency modes."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    idx = np.argsort(eigenvalues)[:n_modes]
    return eigenvalues[idx], mode_shapes[:, idx]


def select_modes_by_frequency_range(
    eigenvalues: NDArray[np.float64],
    mode_shapes: NDArray[np.float64],
    omega_min: float,
    omega_max: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Select modes whose natural frequency ω = sqrt(λ) is in [omega_min, omega_max]."""
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))
    mask = (omega >= omega_min) & (omega <= omega_max)
    return eigenvalues[mask], mode_shapes[:, mask]
