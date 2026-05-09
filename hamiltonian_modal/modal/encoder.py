"""Modal projection encoders.

Physical coordinates (q, v, p) are projected into the truncated modal
subspace spanned by the retained mode shapes Φ:

* **Position**  η = Φᵀ M q  (modal displacement)
* **Velocity**  η̇ = Φᵀ M v  (modal velocity)
* **Momentum**  μ = Φᵀ p    (modal momentum, conjugate to η)

When only *k < n_dof* modes are retained the full-space signal is first
approximated by its projection onto the modal subspace.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hamiltonian_modal.modal.decomposition import ModalBasis

__all__ = [
    "encode_position",
    "encode_velocity",
    "encode_momentum",
]


def _check_vector(x: NDArray[np.float64], expected_size: int, name: str) -> None:
    if x.shape != (expected_size,):
        raise ValueError(
            f"{name} must have shape ({expected_size},); got {x.shape}."
        )


def encode_position(
    basis: ModalBasis,
    M: ArrayLike,
    q: ArrayLike,
) -> NDArray[np.float64]:
    """Project joint-space position *q* into modal coordinates η = Φᵀ M q.

    Parameters
    ----------
    basis:
        Modal basis produced by :func:`~hamiltonian_modal.modal.decomposition.compute_modal_basis`.
    M:
        Mass matrix M(q), shape ``(n_dof, n_dof)``.
    q:
        Joint configuration, shape ``(n_dof,)``.

    Returns
    -------
    NDArray[np.float64]
        Modal displacement η, shape ``(n_modes,)``.
    """
    M_arr = np.asarray(M, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
    _check_vector(q_arr, basis.n_dof, "q")
    return basis.mode_shapes.T @ (M_arr @ q_arr)


def encode_velocity(
    basis: ModalBasis,
    M: ArrayLike,
    v: ArrayLike,
) -> NDArray[np.float64]:
    """Project joint-space velocity *v* into modal velocity η̇ = Φᵀ M v.

    Parameters
    ----------
    basis:
        Modal basis.
    M:
        Mass matrix M(q), shape ``(n_dof, n_dof)``.
    v:
        Joint velocity, shape ``(n_dof,)``.

    Returns
    -------
    NDArray[np.float64]
        Modal velocity η̇, shape ``(n_modes,)``.
    """
    M_arr = np.asarray(M, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
    _check_vector(v_arr, basis.n_dof, "v")
    return basis.mode_shapes.T @ (M_arr @ v_arr)


def encode_momentum(
    basis: ModalBasis,
    p: ArrayLike,
) -> NDArray[np.float64]:
    """Project generalised momentum *p* into modal momentum μ = Φᵀ p.

    Parameters
    ----------
    basis:
        Modal basis.
    p:
        Generalised momentum p = M v, shape ``(n_dof,)``.

    Returns
    -------
    NDArray[np.float64]
        Modal momentum μ, shape ``(n_modes,)``.
    """
    p_arr = np.asarray(p, dtype=np.float64).reshape(-1)
    _check_vector(p_arr, basis.n_dof, "p")
    return basis.mode_shapes.T @ p_arr
