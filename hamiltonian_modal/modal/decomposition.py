"""Modal mass/stiffness assembly and eigendecomposition interfaces.

The *generalised eigenvalue problem* (GEP)

.. math::

    K \\Phi = M \\Phi \\Lambda

is solved using a Cholesky-based reduction to a standard symmetric eigenvalue
problem (no external dependency beyond NumPy):

1. Cholesky factor  M = L Lᵀ
2. Solve  K' = L⁻¹ K L⁻ᵀ  (standard symmetric problem)
3. Eigendecompose  K' Ψ = Ψ Λ  (Ψ is orthonormal w.r.t. identity)
4. Mode shapes  Φ = L⁻ᵀ Ψ  (M-orthonormal: Φᵀ M Φ = I)

Natural frequencies are  ω_i = √λ_i  [rad s⁻¹]; imaginary ω indicates an
unstable equilibrium (negative eigenvalue).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ModalBasis",
    "compute_modal_basis",
    "is_m_orthonormal",
]


@dataclass
class ModalBasis:
    """Container for the result of a modal eigendecomposition.

    Attributes
    ----------
    mode_shapes : NDArray[np.float64]
        Matrix Φ of shape ``(n_dof, n_modes)`` whose columns are the
        M-orthonormal mode shapes (Φᵀ M Φ = I).
    eigenvalues : NDArray[np.float64]
        Diagonal of Λ, shape ``(n_modes,)``.  Non-negative entries correspond
        to stable modes; negative entries indicate instability.
    frequencies : NDArray[np.float64]
        Natural frequencies ω_i = √|λ_i| [rad s⁻¹], shape ``(n_modes,)``.
        Sign convention: positive when λ_i ≥ 0, negative when λ_i < 0.
    n_dof : int
        Full number of degrees of freedom.
    n_modes : int
        Number of retained modes (≤ n_dof).
    """

    mode_shapes: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    n_dof: int
    n_modes: int


def compute_modal_basis(
    M: ArrayLike,
    K: ArrayLike,
    n_modes: int | None = None,
) -> ModalBasis:
    """Compute the modal basis by solving the generalised eigenvalue problem.

    Solves K Φ = M Φ Λ via a Cholesky reduction to a standard symmetric
    eigenproblem.  Eigenvalues are returned in ascending order so the first
    ``n_modes`` columns of Φ correspond to the *lowest-frequency* (most
    energetic) modes.

    Parameters
    ----------
    M:
        Symmetric positive-definite mass matrix, shape ``(n, n)``.
    K:
        Symmetric stiffness matrix, shape ``(n, n)``.  Need not be positive
        definite.
    n_modes:
        Number of modes to retain.  Defaults to all *n* modes.

    Returns
    -------
    ModalBasis
        Dataclass with M-orthonormal mode shapes and associated eigenvalues /
        frequencies.

    Raises
    ------
    ValueError
        If *M* or *K* are not square, not the same size, or *n_modes* is out
        of range.
    np.linalg.LinAlgError
        If *M* is not positive definite (Cholesky fails).
    """
    M_arr = np.asarray(M, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)

    if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
        raise ValueError(f"M must be a square 2-D array; got shape {M_arr.shape}.")
    if K_arr.shape != M_arr.shape:
        raise ValueError(
            f"K shape {K_arr.shape} does not match M shape {M_arr.shape}."
        )
    n = M_arr.shape[0]
    if n_modes is None:
        n_modes = n
    if not (1 <= n_modes <= n):
        raise ValueError(f"n_modes must be in [1, {n}]; got {n_modes}.")

    # Symmetrise inputs to remove numerical asymmetry
    M_sym = 0.5 * (M_arr + M_arr.T)
    K_sym = 0.5 * (K_arr + K_arr.T)

    # Cholesky decomposition: M = L Lᵀ  (L lower triangular)
    L = np.linalg.cholesky(M_sym)
    L_inv = np.linalg.inv(L)

    # Reduce to standard symmetric eigenproblem: K' = L⁻¹ K L⁻ᵀ
    K_prime = L_inv @ K_sym @ L_inv.T

    # Solve standard eigenproblem (returns eigenvalues in ascending order)
    eigenvalues_all, Psi_all = np.linalg.eigh(K_prime)

    # Select the first n_modes (lowest eigenvalues)
    eigenvalues = eigenvalues_all[:n_modes]
    Psi = Psi_all[:, :n_modes]

    # Back-transform: Φ = L⁻ᵀ Ψ
    Phi = L_inv.T @ Psi

    # Natural frequencies (signed: negative for unstable modes)
    signs = np.where(eigenvalues >= 0, 1.0, -1.0)
    frequencies = signs * np.sqrt(np.abs(eigenvalues))

    return ModalBasis(
        mode_shapes=Phi,
        eigenvalues=eigenvalues,
        frequencies=frequencies,
        n_dof=n,
        n_modes=n_modes,
    )


def is_m_orthonormal(
    basis: ModalBasis,
    M: ArrayLike,
    atol: float = 1e-8,
) -> bool:
    """Check whether Φᵀ M Φ ≈ I (M-orthonormality of the mode shapes).

    Parameters
    ----------
    basis:
        :class:`ModalBasis` whose *mode_shapes* are to be checked.
    M:
        Mass matrix used in the eigendecomposition, shape ``(n_dof, n_dof)``.
    atol:
        Absolute tolerance for the identity comparison.

    Returns
    -------
    bool
        ``True`` when Φᵀ M Φ is within *atol* of the identity matrix.
    """
    M_arr = np.asarray(M, dtype=np.float64)
    Phi = basis.mode_shapes
    gram = Phi.T @ M_arr @ Phi
    return bool(np.allclose(gram, np.eye(basis.n_modes), atol=atol))
