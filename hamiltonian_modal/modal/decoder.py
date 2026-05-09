"""Modal reconstruction decoders.

Reconstruct full joint-space signals from truncated modal coordinates:

* **Position**  q ≈ Φ η           (modal → joint displacement)
* **Velocity**  v ≈ Φ η̇          (modal → joint velocity)
* **Momentum**  p = M Φ μ         (modal → generalised momentum)

The reconstruction is exact when all *n_dof* modes are retained; with a
truncated basis (n_modes < n_dof) it is the best rank-*k* approximation in
the M-norm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hamiltonian_modal.modal.decomposition import ModalBasis

__all__ = [
    "decode_position",
    "decode_velocity",
    "decode_momentum",
]


def _check_modal_vector(x: NDArray[np.float64], basis: ModalBasis, name: str) -> None:
    if x.shape != (basis.n_modes,):
        raise ValueError(
            f"{name} must have shape ({basis.n_modes},); got {x.shape}."
        )


def decode_position(
    basis: ModalBasis,
    eta: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct joint-space position from modal displacement η.

    Computes q ≈ Φ η.

    Parameters
    ----------
    basis:
        Modal basis produced by :func:`~hamiltonian_modal.modal.decomposition.compute_modal_basis`.
    eta:
        Modal displacement η, shape ``(n_modes,)``.

    Returns
    -------
    NDArray[np.float64]
        Approximate joint configuration, shape ``(n_dof,)``.
    """
    eta_arr = np.asarray(eta, dtype=np.float64).reshape(-1)
    _check_modal_vector(eta_arr, basis, "eta")
    return basis.mode_shapes @ eta_arr


def decode_velocity(
    basis: ModalBasis,
    eta_dot: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct joint-space velocity from modal velocity η̇.

    Computes v ≈ Φ η̇.

    Parameters
    ----------
    basis:
        Modal basis.
    eta_dot:
        Modal velocity η̇, shape ``(n_modes,)``.

    Returns
    -------
    NDArray[np.float64]
        Approximate joint velocity, shape ``(n_dof,)``.
    """
    eta_dot_arr = np.asarray(eta_dot, dtype=np.float64).reshape(-1)
    _check_modal_vector(eta_dot_arr, basis, "eta_dot")
    return basis.mode_shapes @ eta_dot_arr


def decode_momentum(
    basis: ModalBasis,
    mu: ArrayLike,
    M: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct generalised momentum from modal momentum μ.

    Computes p = M Φ μ.

    Parameters
    ----------
    basis:
        Modal basis.
    mu:
        Modal momentum μ, shape ``(n_modes,)``.
    M:
        Mass matrix, shape ``(n_dof, n_dof)``.

    Returns
    -------
    NDArray[np.float64]
        Generalised momentum p, shape ``(n_dof,)``.
    """
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    M_arr = np.asarray(M, dtype=np.float64)
    _check_modal_vector(mu_arr, basis, "mu")
    return M_arr @ (basis.mode_shapes @ mu_arr)
