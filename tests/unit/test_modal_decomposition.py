"""Unit tests for the Phase 2 modal decomposition layer.

Tests cover:

* :mod:`hamiltonian_modal.modal.stiffness`
* :mod:`hamiltonian_modal.modal.decomposition`
* :mod:`hamiltonian_modal.modal.encoder`
* :mod:`hamiltonian_modal.modal.decoder`
* :mod:`hamiltonian_modal.modal.basis_switching`
* Package-level ``hamiltonian_modal.modal`` re-exports
"""

from __future__ import annotations

import numpy as np
import pytest

import hamiltonian_modal.modal as hm_modal
from hamiltonian_modal.modal.basis_switching import (
    select_low_frequency_modes,
    select_modes,
    select_modes_by_frequency_range,
)
from hamiltonian_modal.modal.decoder import decode_momentum, decode_position, decode_velocity
from hamiltonian_modal.modal.decomposition import (
    ModalBasis,
    compute_modal_basis,
    is_m_orthonormal,
)
from hamiltonian_modal.modal.encoder import encode_momentum, encode_position, encode_velocity
from hamiltonian_modal.modal.stiffness import diagonal_stiffness, stiffness_from_gravity_hessian


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_spd(n: int, seed: int = 0) -> np.ndarray:
    """Return a random symmetric positive-definite matrix of size (n, n)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n) * n  # add nI so eigenvalues > n


def _simple_system(n: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (M, K) for a small test system."""
    M = _make_spd(n, seed=seed)
    K = _make_spd(n, seed=seed + 1)
    return M, K


# ---------------------------------------------------------------------------
# stiffness.py
# ---------------------------------------------------------------------------


class TestDiagonalStiffness:
    def test_scalar_broadcast(self) -> None:
        K = diagonal_stiffness(4, k_values=2.5)
        assert K.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(K), [2.5, 2.5, 2.5, 2.5])
        np.testing.assert_array_equal(K - np.diag(np.diag(K)), 0.0)

    def test_array_k_values(self) -> None:
        k = [1.0, 2.0, 3.0]
        K = diagonal_stiffness(3, k_values=k)
        np.testing.assert_array_equal(np.diag(K), k)

    def test_zero_stiffness_allowed(self) -> None:
        K = diagonal_stiffness(2, k_values=0.0)
        np.testing.assert_array_equal(K, np.zeros((2, 2)))

    def test_rejects_negative_stiffness(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            diagonal_stiffness(3, k_values=[-1.0, 2.0, 3.0])

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="does not match n_dof"):
            diagonal_stiffness(4, k_values=[1.0, 2.0])

    def test_dtype_is_float64(self) -> None:
        K = diagonal_stiffness(2, k_values=1)
        assert K.dtype == np.float64


class TestStiffnessFromGravityHessian:
    def test_recovers_linear_system_stiffness(self) -> None:
        # For a linear system where g(q) = K0 q, the Jacobian should equal K0.
        K0 = np.array([[4.0, 0.5], [0.5, 3.0]])

        def gravity_fn(q: np.ndarray) -> np.ndarray:
            return K0 @ q

        q_eq = np.zeros(2)
        K_est = stiffness_from_gravity_hessian(gravity_fn, q_eq, eps=1e-5)
        np.testing.assert_allclose(K_est, K0, atol=1e-6)

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(7)
        A = rng.standard_normal((3, 3))

        def gravity_fn(q: np.ndarray) -> np.ndarray:
            return A @ q

        K_est = stiffness_from_gravity_hessian(gravity_fn, np.zeros(3))
        np.testing.assert_allclose(K_est, K_est.T, atol=1e-10)

    def test_output_shape(self) -> None:
        def gravity_fn(q: np.ndarray) -> np.ndarray:
            return q

        K_est = stiffness_from_gravity_hessian(gravity_fn, np.zeros(5))
        assert K_est.shape == (5, 5)


# ---------------------------------------------------------------------------
# decomposition.py
# ---------------------------------------------------------------------------


class TestComputeModalBasis:
    def test_returns_modal_basis_type(self) -> None:
        M, K = _simple_system()
        basis = compute_modal_basis(M, K)
        assert isinstance(basis, ModalBasis)

    def test_default_n_modes_is_n_dof(self) -> None:
        n = 5
        M, K = _simple_system(n)
        basis = compute_modal_basis(M, K)
        assert basis.n_modes == n
        assert basis.n_dof == n

    def test_truncated_n_modes(self) -> None:
        M, K = _simple_system(6)
        basis = compute_modal_basis(M, K, n_modes=3)
        assert basis.n_modes == 3
        assert basis.mode_shapes.shape == (6, 3)

    def test_mode_shapes_are_m_orthonormal(self) -> None:
        M, K = _simple_system(5)
        basis = compute_modal_basis(M, K)
        gram = basis.mode_shapes.T @ M @ basis.mode_shapes
        np.testing.assert_allclose(gram, np.eye(basis.n_modes), atol=1e-10)

    def test_kphi_equals_mphi_lambda(self) -> None:
        """Verify K Φ ≈ M Φ Λ (the GEP residual)."""
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        lhs = K @ basis.mode_shapes
        rhs = M @ basis.mode_shapes @ np.diag(basis.eigenvalues)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_eigenvalues_ascending(self) -> None:
        M, K = _simple_system(6)
        basis = compute_modal_basis(M, K)
        assert np.all(np.diff(basis.eigenvalues) >= 0)

    def test_frequencies_sign_convention(self) -> None:
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        # For a positive-definite K, all frequencies should be positive
        assert np.all(basis.frequencies > 0)

    def test_frequency_equals_sqrt_eigenvalue(self) -> None:
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        np.testing.assert_allclose(
            basis.frequencies, np.sqrt(basis.eigenvalues), atol=1e-12
        )

    def test_rejects_non_square_M(self) -> None:
        with pytest.raises(ValueError, match="square"):
            compute_modal_basis(np.ones((3, 4)), np.eye(4))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            compute_modal_basis(np.eye(3), np.eye(4))

    def test_rejects_out_of_range_n_modes(self) -> None:
        M, K = _simple_system(4)
        with pytest.raises(ValueError, match="n_modes must be in"):
            compute_modal_basis(M, K, n_modes=5)
        with pytest.raises(ValueError, match="n_modes must be in"):
            compute_modal_basis(M, K, n_modes=0)

    def test_rejects_non_pd_mass_matrix(self) -> None:
        M_singular = np.zeros((3, 3))
        K = np.eye(3)
        with pytest.raises(np.linalg.LinAlgError):
            compute_modal_basis(M_singular, K)


class TestIsMOrthonormal:
    def test_true_for_correct_basis(self) -> None:
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        assert is_m_orthonormal(basis, M) is True

    def test_false_for_scaled_modes(self) -> None:
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        bad_basis = ModalBasis(
            mode_shapes=basis.mode_shapes * 2.0,
            eigenvalues=basis.eigenvalues,
            frequencies=basis.frequencies,
            n_dof=basis.n_dof,
            n_modes=basis.n_modes,
        )
        assert is_m_orthonormal(bad_basis, M) is False

    def test_tight_atol_may_fail(self) -> None:
        M, K = _simple_system(4)
        basis = compute_modal_basis(M, K)
        # With an extremely tight tolerance, floating-point error may trigger False
        result = is_m_orthonormal(basis, M, atol=1e-15)
        # We don't assert a specific value but just check it returns a bool
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# encoder.py
# ---------------------------------------------------------------------------


class TestEncoder:
    def setup_method(self) -> None:
        M, K = _simple_system(4)
        self.M = M
        self.basis = compute_modal_basis(M, K)

    def test_encode_position_shape(self) -> None:
        q = np.ones(4)
        eta = encode_position(self.basis, self.M, q)
        assert eta.shape == (self.basis.n_modes,)

    def test_encode_velocity_shape(self) -> None:
        v = np.ones(4)
        eta_dot = encode_velocity(self.basis, self.M, v)
        assert eta_dot.shape == (self.basis.n_modes,)

    def test_encode_momentum_shape(self) -> None:
        p = np.ones(4)
        mu = encode_momentum(self.basis, p)
        assert mu.shape == (self.basis.n_modes,)

    def test_encode_position_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            encode_position(self.basis, self.M, np.ones(5))

    def test_encode_velocity_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            encode_velocity(self.basis, self.M, np.ones(3))

    def test_encode_momentum_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            encode_momentum(self.basis, np.ones(3))

    def test_encode_position_is_linear(self) -> None:
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.0, 1.0, 0.0, 0.0])
        alpha = 2.5
        eta1 = encode_position(self.basis, self.M, q1)
        eta2 = encode_position(self.basis, self.M, q2)
        eta_combined = encode_position(self.basis, self.M, alpha * q1 + q2)
        np.testing.assert_allclose(eta_combined, alpha * eta1 + eta2, atol=1e-12)


# ---------------------------------------------------------------------------
# decoder.py
# ---------------------------------------------------------------------------


class TestDecoder:
    def setup_method(self) -> None:
        M, K = _simple_system(4)
        self.M = M
        self.basis = compute_modal_basis(M, K)

    def test_decode_position_shape(self) -> None:
        eta = np.ones(self.basis.n_modes)
        q = decode_position(self.basis, eta)
        assert q.shape == (4,)

    def test_decode_velocity_shape(self) -> None:
        eta_dot = np.ones(self.basis.n_modes)
        v = decode_velocity(self.basis, eta_dot)
        assert v.shape == (4,)

    def test_decode_momentum_shape(self) -> None:
        mu = np.ones(self.basis.n_modes)
        p = decode_momentum(self.basis, mu, self.M)
        assert p.shape == (4,)

    def test_decode_position_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            decode_position(self.basis, np.ones(self.basis.n_modes + 1))

    def test_decode_velocity_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            decode_velocity(self.basis, np.ones(0))

    def test_decode_momentum_wrong_size(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            decode_momentum(self.basis, np.ones(1), self.M)


# ---------------------------------------------------------------------------
# Encoder-Decoder roundtrip
# ---------------------------------------------------------------------------


class TestEncoderDecoderRoundtrip:
    """With a full (non-truncated) basis, encode ∘ decode should be identity."""

    def setup_method(self) -> None:
        M, K = _simple_system(4)
        self.M = M
        self.basis = compute_modal_basis(M, K)  # full basis

    def test_position_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        # Generate q in the column space of Φ (i.e. q = Φ η₀ for some η₀)
        eta0 = rng.standard_normal(self.basis.n_modes)
        q = decode_position(self.basis, eta0)
        eta_recovered = encode_position(self.basis, self.M, q)
        np.testing.assert_allclose(eta_recovered, eta0, atol=1e-10)

    def test_velocity_roundtrip(self) -> None:
        rng = np.random.default_rng(1)
        eta_dot0 = rng.standard_normal(self.basis.n_modes)
        v = decode_velocity(self.basis, eta_dot0)
        eta_dot_recovered = encode_velocity(self.basis, self.M, v)
        np.testing.assert_allclose(eta_dot_recovered, eta_dot0, atol=1e-10)

    def test_momentum_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        mu0 = rng.standard_normal(self.basis.n_modes)
        p = decode_momentum(self.basis, mu0, self.M)
        mu_recovered = encode_momentum(self.basis, p)
        np.testing.assert_allclose(mu_recovered, mu0, atol=1e-10)


# ---------------------------------------------------------------------------
# basis_switching.py
# ---------------------------------------------------------------------------


class TestSelectModes:
    def setup_method(self) -> None:
        M, K = _simple_system(6)
        self.basis = compute_modal_basis(M, K)

    def test_select_two_modes(self) -> None:
        sub = select_modes(self.basis, [0, 2])
        assert sub.n_modes == 2
        assert sub.mode_shapes.shape == (6, 2)
        np.testing.assert_array_equal(sub.eigenvalues, self.basis.eigenvalues[[0, 2]])

    def test_single_mode(self) -> None:
        sub = select_modes(self.basis, [3])
        assert sub.n_modes == 1

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            select_modes(self.basis, [])

    def test_rejects_out_of_range_index(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            select_modes(self.basis, [0, 99])

    def test_n_dof_preserved(self) -> None:
        sub = select_modes(self.basis, [1, 4])
        assert sub.n_dof == self.basis.n_dof


class TestSelectLowFrequencyModes:
    def setup_method(self) -> None:
        M, K = _simple_system(6)
        self.basis = compute_modal_basis(M, K)

    def test_returns_correct_count(self) -> None:
        sub = select_low_frequency_modes(self.basis, 3)
        assert sub.n_modes == 3

    def test_modes_have_lowest_frequencies(self) -> None:
        n = 3
        sub = select_low_frequency_modes(self.basis, n)
        sorted_freq = np.sort(np.abs(self.basis.frequencies))
        np.testing.assert_array_equal(
            np.sort(np.abs(sub.frequencies)), sorted_freq[:n]
        )

    def test_rejects_n_modes_zero(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be in"):
            select_low_frequency_modes(self.basis, 0)

    def test_rejects_n_modes_too_large(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be in"):
            select_low_frequency_modes(self.basis, self.basis.n_modes + 1)


class TestSelectModesByFrequencyRange:
    def setup_method(self) -> None:
        M, K = _simple_system(6)
        self.basis = compute_modal_basis(M, K)

    def test_full_range_returns_all(self) -> None:
        f_max = float(np.max(np.abs(self.basis.frequencies))) + 1.0
        sub = select_modes_by_frequency_range(self.basis, 0.0, f_max)
        assert sub.n_modes == self.basis.n_modes

    def test_no_modes_in_range_raises(self) -> None:
        with pytest.raises(ValueError, match="No modes found"):
            select_modes_by_frequency_range(self.basis, 1e9, 2e9)

    def test_rejects_f_min_negative(self) -> None:
        with pytest.raises(ValueError, match="f_min must be"):
            select_modes_by_frequency_range(self.basis, -1.0, 10.0)

    def test_rejects_f_max_less_than_f_min(self) -> None:
        with pytest.raises(ValueError, match="f_max"):
            select_modes_by_frequency_range(self.basis, 5.0, 1.0)

    def test_selected_frequencies_within_range(self) -> None:
        f_min = float(np.min(np.abs(self.basis.frequencies)))
        f_max = float(np.median(np.abs(self.basis.frequencies)))
        if f_min >= f_max:
            pytest.skip("degenerate frequency distribution; skipping range test")
        sub = select_modes_by_frequency_range(self.basis, f_min, f_max)
        assert np.all(np.abs(sub.frequencies) >= f_min)
        assert np.all(np.abs(sub.frequencies) <= f_max)


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------


class TestPackageReexports:
    def test_all_public_names_importable(self) -> None:
        expected = [
            "ModalBasis",
            "compute_modal_basis",
            "is_m_orthonormal",
            "diagonal_stiffness",
            "stiffness_from_gravity_hessian",
            "encode_position",
            "encode_velocity",
            "encode_momentum",
            "decode_position",
            "decode_velocity",
            "decode_momentum",
            "select_modes",
            "select_low_frequency_modes",
            "select_modes_by_frequency_range",
        ]
        for name in expected:
            assert hasattr(hm_modal, name), f"hamiltonian_modal.modal.{name} not exported"
