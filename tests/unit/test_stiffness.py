"""Unit tests for hamiltonian_modal.modal.stiffness."""

from __future__ import annotations

import numpy as np
import pytest

from hamiltonian_modal.modal.stiffness import (
    assemble_stiffness_matrix,
    diagonal_stiffness,
    stiffness_from_gravity_hessian,
)


class TestAssembleStiffnessMatrix:
    def test_assemble_stiffness_symmetric(self, standing_config: np.ndarray) -> None:
        """K must be symmetric (K == K.T)."""
        K = assemble_stiffness_matrix(model=None, q_ref=standing_config)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_assemble_stiffness_psd(self, standing_config: np.ndarray) -> None:
        """All eigenvalues of K must be >= 0 (positive semi-definite)."""
        K = assemble_stiffness_matrix(model=None, q_ref=standing_config)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-12), (
            f"K has negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
        )

    def test_assemble_stiffness_shape(self, standing_config: np.ndarray, n_dof: int) -> None:
        """K must be square with side length equal to n_dof."""
        K = assemble_stiffness_matrix(model=None, q_ref=standing_config)
        assert K.shape == (n_dof, n_dof)

    def test_assemble_stiffness_custom_pd_gains(self, standing_config: np.ndarray) -> None:
        """Custom PD gains must be reflected in the diagonal of K."""
        n = len(standing_config)
        custom_gains = np.linspace(50.0, 400.0, n)
        K = assemble_stiffness_matrix(
            model=None, q_ref=standing_config, joint_pd_gains=custom_gains
        )
        # The diagonal must be at least as large as the PD gain
        diag = np.diag(K)
        assert np.all(diag >= custom_gains - 1e-9), (
            "Diagonal of K must dominate PD gains"
        )

    def test_assemble_stiffness_wrong_gain_shape_raises(
        self, standing_config: np.ndarray
    ) -> None:
        """Mismatched joint_pd_gains shape must raise ValueError."""
        with pytest.raises(ValueError):
            assemble_stiffness_matrix(
                model=None,
                q_ref=standing_config,
                joint_pd_gains=np.ones(10),  # wrong size
            )

    def test_assemble_stiffness_default_vs_custom_material(
        self, standing_config: np.ndarray
    ) -> None:
        """Higher Young's modulus must produce a stiffer matrix."""
        K_low = assemble_stiffness_matrix(
            model=None,
            q_ref=standing_config,
            link_material={"E": 10e9, "rho": 2700.0},
        )
        K_high = assemble_stiffness_matrix(
            model=None,
            q_ref=standing_config,
            link_material={"E": 200e9, "rho": 7800.0},
        )
        # The trace (sum of diagonal = sum of per-DOF stiffness) must be larger
        assert np.trace(K_high) > np.trace(K_low)

    def test_assemble_stiffness_different_configs_produce_different_K(
        self,
    ) -> None:
        """Different gearbox compliance values must produce different matrices."""
        q = np.zeros(23)
        K1 = assemble_stiffness_matrix(model=None, q_ref=q, gearbox_compliance=1000.0)
        K2 = assemble_stiffness_matrix(model=None, q_ref=q, gearbox_compliance=10000.0)
        assert not np.allclose(K1, K2)


class TestDiagonalStiffness:
    def test_diagonal_stiffness_shape(self) -> None:
        """Output shape must be (n_dof, n_dof)."""
        n = 10
        K = diagonal_stiffness(n, np.ones(n) * 200.0)
        assert K.shape == (n, n)

    def test_diagonal_stiffness_values(self) -> None:
        """Diagonal values must match the input k_values exactly."""
        k_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        K = diagonal_stiffness(5, k_vals)
        np.testing.assert_array_equal(np.diag(K), k_vals)

    def test_diagonal_stiffness_off_diagonal_zero(self) -> None:
        """All off-diagonal entries must be zero."""
        k_vals = np.array([100.0, 200.0, 300.0])
        K = diagonal_stiffness(3, k_vals)
        off_diag = K - np.diag(np.diag(K))
        np.testing.assert_array_equal(off_diag, np.zeros((3, 3)))

    def test_diagonal_stiffness_wrong_size_raises(self) -> None:
        """Mismatched n_dof and k_values length must raise ValueError."""
        with pytest.raises(ValueError):
            diagonal_stiffness(5, np.ones(4))

    def test_diagonal_stiffness_single_dof(self) -> None:
        """Must work correctly for a 1-DOF system."""
        K = diagonal_stiffness(1, np.array([999.9]))
        assert K.shape == (1, 1)
        assert K[0, 0] == pytest.approx(999.9)


class TestStiffnessFromGravityHessian:
    def _harmonic_potential(self, q: np.ndarray) -> float:
        """V(q) = 0.5 * sum(k_i * q_i^2), analytical Hessian = diag(k)."""
        k = np.array([100.0, 200.0, 500.0])
        return float(0.5 * np.dot(k, q**2))

    def test_stiffness_from_gravity_hessian_symmetric(self) -> None:
        """Hessian must be symmetric."""
        q_eq = np.zeros(3)
        K = stiffness_from_gravity_hessian(self._harmonic_potential, q_eq, eps=1e-5)
        np.testing.assert_allclose(K, K.T, atol=1e-8)

    def test_stiffness_from_gravity_hessian_recovers_analytical(self) -> None:
        """FD Hessian of a quadratic potential must match analytical Hessian."""
        q_eq = np.zeros(3)
        K = stiffness_from_gravity_hessian(self._harmonic_potential, q_eq, eps=1e-5)
        expected = np.diag([100.0, 200.0, 500.0])
        np.testing.assert_allclose(K, expected, atol=1e-4)

    def test_stiffness_from_gravity_hessian_psd_for_convex(self) -> None:
        """Hessian of a convex potential must be PSD."""
        q_eq = np.zeros(4)

        def convex_pot(q: np.ndarray) -> float:
            # V = sum(q_i^4) — convex
            return float(np.sum(q**4))

        K = stiffness_from_gravity_hessian(convex_pot, q_eq, eps=1e-4)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8), "Hessian of convex function must be PSD"

    def test_stiffness_from_gravity_hessian_shape(self) -> None:
        """Output shape must equal (n, n) where n = len(q_eq)."""
        n = 6
        q_eq = np.zeros(n)

        def pot(q: np.ndarray) -> float:
            return float(np.sum(q**2))

        K = stiffness_from_gravity_hessian(pot, q_eq)
        assert K.shape == (n, n)
