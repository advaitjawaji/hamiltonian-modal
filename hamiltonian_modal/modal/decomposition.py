"""Modal decomposition via generalized eigenvalue problem.

Given a mass matrix M and stiffness matrix K, solves the generalized
eigenvalue problem::

    K φ = ω² M φ

using SciPy's ``eigh`` (symmetric/Hermitian solver).  The returned mode
shapes are M-orthonormal: Φ^T M Φ = I.

Functions
---------
compute_modal_decomposition
    Solve the generalized EVP and return eigenvalues and mode shapes.
compute_modal_basis
    Alias for ``compute_modal_decomposition`` (backward compatibility).
is_m_orthonormal
    Check whether a set of mode shapes satisfies Φ^T M Φ ≈ I.
precompute_g1_modes
    Precompute and cache mode shapes for a list of reference configs.

References
----------
.. [1] Clough, R. W., & Penzien, J. (2003). *Dynamics of Structures*
       (3rd ed.). Computers & Structures, Inc.
.. [2] Bathe, K.-J. (1996). *Finite Element Procedures*. Prentice Hall.

Examples
--------
>>> import numpy as np
>>> from hamiltonian_modal.modal.decomposition import compute_modal_decomposition
>>> M = np.eye(5) * 2.0
>>> K = np.diag([0.0, 100.0, 500.0, 1000.0, 5000.0])
>>> eigenvalues, mode_shapes = compute_modal_decomposition(M, K, n_modes=5)
>>> eigenvalues.shape
(5,)
>>> mode_shapes.shape
(5, 5)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


def compute_modal_decomposition(
    M: NDArray[np.float64],
    K: NDArray[np.float64],
    n_modes: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve the generalized eigenvalue problem K φ = ω² M φ.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Symmetric positive-definite mass matrix.
    K : np.ndarray, shape (n, n)
        Symmetric positive-semidefinite stiffness matrix.
    n_modes : int, optional
        Number of modes to compute.  Clamped to ``n`` if larger.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_modes,)
        Squared natural frequencies ω² in ascending order (rad²/s²).
        Small negative values due to floating-point noise are clamped to 0.
    mode_shapes : np.ndarray, shape (n, n_modes)
        M-orthonormal mode shapes: columns φ_i satisfy
        ``φ_i^T M φ_j = δ_ij`` and ``φ_i^T K φ_j = ω_i² δ_ij``.

    Raises
    ------
    ValueError
        If M or K are not square with matching dimensions, or if
        ``n_modes`` is not a positive integer.
    numpy.linalg.LinAlgError
        If M is not positive definite (i.e. SciPy's ``eigh`` fails).

    Notes
    -----
    SciPy's ``eigh(K, M, subset_by_index=[0, n_modes-1])`` uses the
    divide-and-conquer algorithm which is O(n³) but numerically stable
    for small-to-medium n (up to ~500 DOFs for the G1 use-case).

    For a floating-base robot model the first 6 eigenvalues correspond
    to rigid-body (zero-frequency) modes.  These are preserved in the
    output and can be identified as eigenvalues ≈ 0.
    """
    M = np.asarray(M, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be a square 2-D array, got shape {M.shape}")
    if K.shape != M.shape:
        raise ValueError(
            f"K and M must have the same shape; got K={K.shape}, M={M.shape}"
        )
    if not isinstance(n_modes, int) or n_modes < 1:
        raise ValueError(f"n_modes must be a positive integer, got {n_modes!r}")

    n = M.shape[0]
    n_modes = min(n_modes, n)

    logger.debug(
        "Computing modal decomposition: n=%d, n_modes=%d", n, n_modes
    )

    # scipy eigh solves K v = w * M v, returning eigenvalues in ascending order
    eigenvalues, mode_shapes = eigh(K, M, subset_by_index=[0, n_modes - 1])

    # Clamp small negative eigenvalues (numerical noise in near-zero modes)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    logger.info(
        "Modal decomposition complete: ω² range [%.4g, %.4g] rad²/s²",
        eigenvalues[0],
        eigenvalues[-1],
    )
    return eigenvalues, mode_shapes


def compute_modal_basis(
    M: NDArray[np.float64],
    K: NDArray[np.float64],
    n_modes: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Alias for :func:`compute_modal_decomposition`.

    Kept for backward compatibility with code written before the function
    was renamed.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        Symmetric positive-definite mass matrix.
    K : np.ndarray, shape (n, n)
        Symmetric positive-semidefinite stiffness matrix.
    n_modes : int, optional
        Number of modes to retain.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_modes,)
    mode_shapes : np.ndarray, shape (n, n_modes)

    See Also
    --------
    compute_modal_decomposition
    """
    return compute_modal_decomposition(M, K, n_modes)


def is_m_orthonormal(
    mode_shapes: NDArray[np.float64],
    M: NDArray[np.float64],
    atol: float = 1e-8,
) -> bool:
    """Check whether Φ^T M Φ ≈ I (M-orthonormality).

    Parameters
    ----------
    mode_shapes : np.ndarray, shape (n, k)
        Matrix of mode shape columns Φ = [φ_1, …, φ_k].
    M : np.ndarray, shape (n, n)
        Mass matrix.
    atol : float, optional
        Absolute tolerance for ``numpy.allclose``.

    Returns
    -------
    bool
        ``True`` if ``Φ^T M Φ`` is close to the identity matrix within
        ``atol``, ``False`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from hamiltonian_modal.modal.decomposition import (
    ...     compute_modal_decomposition, is_m_orthonormal
    ... )
    >>> M = np.eye(4) * 3.0
    >>> K = np.diag([10.0, 50.0, 200.0, 800.0])
    >>> _, Phi = compute_modal_decomposition(M, K, n_modes=4)
    >>> is_m_orthonormal(Phi, M)
    True
    """
    mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    product = mode_shapes.T @ M @ mode_shapes
    expected = np.eye(mode_shapes.shape[1])
    return bool(np.allclose(product, expected, atol=atol))


def precompute_g1_modes(
    reference_configurations: list[tuple[str, NDArray[np.float64]]],
    n_modes: int = 20,
    cache_dir: str = "data/g1_modes",
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Precompute and cache mode shapes at multiple reference configurations.

    For each (name, q_ref) pair, assembles M and K (via Pinocchio if
    available, otherwise uses synthetic placeholders), solves the
    generalized EVP, and saves the result as a ``.npz`` file in
    ``cache_dir``.

    Parameters
    ----------
    reference_configurations : list of (str, np.ndarray) pairs
        Named joint configurations.  Each element is a tuple
        ``(name, q_ref)`` where ``name`` is used as the cache filename
        (without extension) and ``q_ref`` is a 1-D joint configuration
        array.
    n_modes : int, optional
        Number of modes to compute per configuration.
    cache_dir : str, optional
        Directory in which to write ``.npz`` cache files.  Created
        automatically if it does not exist.

    Returns
    -------
    results : dict
        Mapping ``name -> {"eigenvalues": ..., "mode_shapes": ..., "q_ref": ...}``.

    Notes
    -----
    If Pinocchio is not installed, the function falls back to a synthetic
    mass matrix (``M = 5 * I``) and a linearly spaced stiffness diagonal.
    A warning is logged in this case.  The resulting mode shapes are not
    physically meaningful but allow downstream tests to run without
    optional dependencies.
    """
    import pathlib

    from hamiltonian_modal.modal.stiffness import assemble_stiffness_matrix

    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, NDArray[np.float64]]] = {}

    for name, q_ref in reference_configurations:
        q_ref = np.asarray(q_ref, dtype=np.float64)
        n = q_ref.shape[0]

        # Attempt to build physically accurate M and K via Pinocchio
        try:
            from hamiltonian_modal.utils.pinocchio_wrapper import (
                load_g1_model,
                mass_matrix_at,
            )

            model, data = load_g1_model()

            # Construct full Pinocchio configuration (with floating base)
            q_pin = np.zeros(model.nq, dtype=np.float64)
            # Set quaternion w = 1 for floating base (indices 3..6)
            if model.nq >= 7:
                q_pin[6] = 1.0
            actuated_start = model.nq - model.nv + 6  # skip 6-DOF base
            n_actuated = min(n, model.nv - 6)
            q_pin[model.nq - n_actuated :] = q_ref[:n_actuated]

            M = mass_matrix_at(model, data, q_pin)
            K = assemble_stiffness_matrix(model, q_ref[: model.nv])
            # Trim to actuated subspace if needed
            n_act = min(n, M.shape[0])
            M = M[:n_act, :n_act]
            K = K[:n_act, :n_act]

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Pinocchio unavailable for config '%s' (%s); "
                "using synthetic mass matrix.",
                name,
                exc,
            )
            M = np.eye(n, dtype=np.float64) * 5.0
            K = np.diag(np.linspace(100.0, 5000.0, n))

        n_modes_actual = min(n_modes, M.shape[0])
        eigenvalues, mode_shapes = compute_modal_decomposition(M, K, n_modes_actual)

        result: dict[str, NDArray[np.float64]] = {
            "eigenvalues": eigenvalues,
            "mode_shapes": mode_shapes,
            "q_ref": q_ref,
        }
        results[name] = result

        npz_path = cache_path / f"{name}.npz"
        np.savez(str(npz_path), **result)
        logger.info("Saved modes for config '%s' to %s", name, npz_path)

    return results
