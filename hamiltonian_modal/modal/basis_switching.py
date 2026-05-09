"""Configuration-dependent modal basis switching interfaces.

During locomotion a humanoid robot transitions between contact configurations
(standing, walking, jumping …).  Each configuration may have a different
effective stiffness distribution, making a *fixed* global modal basis
sub-optimal.

The helpers in this module let downstream code select a *sub-basis* of the
most relevant modes without re-solving the eigenvalue problem from scratch:

* :func:`select_modes` — pick an arbitrary subset by column index.
* :func:`select_low_frequency_modes` — retain the *k* lowest-frequency modes
  (the default strategy for smooth, energy-efficient motion).
* :func:`select_modes_by_frequency_range` — keep modes whose natural
  frequency |ω| lies in ``[f_min, f_max]`` rad s⁻¹.
"""

from __future__ import annotations

import numpy as np

from hamiltonian_modal.modal.decomposition import ModalBasis

__all__ = [
    "select_modes",
    "select_low_frequency_modes",
    "select_modes_by_frequency_range",
]


def select_modes(basis: ModalBasis, mode_indices: list[int]) -> ModalBasis:
    """Return a new :class:`~hamiltonian_modal.modal.decomposition.ModalBasis`
    containing only the specified columns.

    Parameters
    ----------
    basis:
        Source modal basis with ``n_modes`` modes.
    mode_indices:
        List of column indices to retain.  Each index must be in
        ``[0, basis.n_modes)``.  Duplicates are preserved.

    Returns
    -------
    ModalBasis
        Sub-basis with ``len(mode_indices)`` modes.

    Raises
    ------
    ValueError
        If any index is out of range or *mode_indices* is empty.
    """
    if not mode_indices:
        raise ValueError("mode_indices must not be empty.")
    idx = list(mode_indices)
    for i in idx:
        if not (0 <= i < basis.n_modes):
            raise ValueError(
                f"Mode index {i} is out of range [0, {basis.n_modes})."
            )
    cols = np.array(idx, dtype=int)
    return ModalBasis(
        mode_shapes=basis.mode_shapes[:, cols],
        eigenvalues=basis.eigenvalues[cols],
        frequencies=basis.frequencies[cols],
        n_dof=basis.n_dof,
        n_modes=len(idx),
    )


def select_low_frequency_modes(basis: ModalBasis, n_modes: int) -> ModalBasis:
    """Retain the *n_modes* modes with the smallest |ω| (lowest frequency).

    Parameters
    ----------
    basis:
        Source modal basis.  Modes are assumed to be sorted in ascending order
        of eigenvalue (as produced by
        :func:`~hamiltonian_modal.modal.decomposition.compute_modal_basis`).
    n_modes:
        Number of low-frequency modes to keep.  Must be in
        ``[1, basis.n_modes]``.

    Returns
    -------
    ModalBasis
        Sub-basis with *n_modes* low-frequency modes.

    Raises
    ------
    ValueError
        If *n_modes* is out of range.
    """
    if not (1 <= n_modes <= basis.n_modes):
        raise ValueError(
            f"n_modes must be in [1, {basis.n_modes}]; got {n_modes}."
        )
    # Modes are already sorted by |eigenvalue| ascending after eigh
    order = np.argsort(np.abs(basis.frequencies))
    selected = order[:n_modes].tolist()
    return select_modes(basis, selected)


def select_modes_by_frequency_range(
    basis: ModalBasis,
    f_min: float,
    f_max: float,
) -> ModalBasis:
    """Select modes whose natural frequency |ω| lies in [*f_min*, *f_max*] rad s⁻¹.

    Parameters
    ----------
    basis:
        Source modal basis.
    f_min:
        Lower bound on |ω| (inclusive), rad s⁻¹.  Must be ≥ 0.
    f_max:
        Upper bound on |ω| (inclusive), rad s⁻¹.  Must be ≥ *f_min*.

    Returns
    -------
    ModalBasis
        Sub-basis containing only modes within the frequency band.

    Raises
    ------
    ValueError
        If no modes fall within the specified range, or *f_min* > *f_max*.
    """
    if f_min < 0:
        raise ValueError(f"f_min must be ≥ 0; got {f_min}.")
    if f_max < f_min:
        raise ValueError(f"f_max ({f_max}) must be ≥ f_min ({f_min}).")
    abs_freq = np.abs(basis.frequencies)
    indices = [int(i) for i in np.where((abs_freq >= f_min) & (abs_freq <= f_max))[0]]
    if not indices:
        raise ValueError(
            f"No modes found with |ω| in [{f_min}, {f_max}] rad s⁻¹.  "
            f"Available frequencies: {abs_freq.tolist()}"
        )
    return select_modes(basis, indices)
