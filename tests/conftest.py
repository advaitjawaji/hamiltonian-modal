"""Shared pytest fixtures for the hamiltonian-modal test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def small_mass_matrix() -> np.ndarray:
    """5x5 symmetric positive definite mass matrix for fast tests.

    Constructed as A @ A.T + 5*I to guarantee positive definiteness.
    """
    A = np.random.RandomState(42).randn(5, 5)
    return A @ A.T + 5 * np.eye(5)


@pytest.fixture(scope="session")
def small_stiffness_matrix() -> np.ndarray:
    """5x5 symmetric positive semi-definite stiffness matrix.

    The first eigenvalue is 0 (rigid-body mode); the rest are positive.
    """
    return np.diag([0.0, 100.0, 500.0, 1000.0, 5000.0])


@pytest.fixture(scope="session")
def n_dof() -> int:
    """Number of actuated DOFs for the Unitree G1 humanoid."""
    return 23


@pytest.fixture(scope="session")
def standing_config(n_dof: int) -> np.ndarray:
    """Default standing joint configuration (all zeros)."""
    return np.zeros(n_dof, dtype=np.float64)


@pytest.fixture(scope="session")
def modal_basis(
    small_mass_matrix: np.ndarray,
    small_stiffness_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precomputed modal decomposition for the 5-DOF test system.

    Returns
    -------
    (eigenvalues, mode_shapes) : tuple of np.ndarray
        eigenvalues shape (5,), mode_shapes shape (5, 5).
    """
    from hamiltonian_modal.modal.decomposition import compute_modal_decomposition

    return compute_modal_decomposition(
        small_mass_matrix, small_stiffness_matrix, n_modes=5
    )
