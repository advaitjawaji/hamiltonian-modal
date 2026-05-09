"""Unit tests for hamiltonian_modal.modal.decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from hamiltonian_modal.modal.decomposition import (
    compute_modal_decomposition,
    is_m_orthonormal,
    precompute_g1_modes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spd(n: int, seed: int = 0) -> np.ndarray:
    """Return an n x n symmetric positive-definite matrix."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return A @ A.T + n * np.eye(n)


def _make_spsd(n: int) -> np.ndarray:
    """Return an n x n symmetric positive-semidefinite matrix with 6 zeros."""
    k = np.zeros(n)
    k[6:] = np.linspace(100.0, 5000.0, n - 6)
    return np.diag(k)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeModalDecomposition:
    def test_orthonormality(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """Mode shapes must satisfy Φ^T M Φ = I (M-orthonormality)."""
        eigenvalues, mode_shapes = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=5
        )
        product = mode_shapes.T @ small_mass_matrix @ mode_shapes
        np.testing.assert_allclose(product, np.eye(5), atol=1e-8)

    def test_stiffness_diagonalization(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """Φ^T K Φ must be diagonal with entries equal to eigenvalues."""
        eigenvalues, mode_shapes = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=5
        )
        diagonalized = mode_shapes.T @ small_stiffness_matrix @ mode_shapes
        expected = np.diag(eigenvalues)
        np.testing.assert_allclose(diagonalized, expected, atol=1e-8)

    def test_frequency_ordering(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """Eigenvalues must be in ascending order and all >= 0."""
        eigenvalues, _ = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=5
        )
        assert np.all(eigenvalues >= 0.0), "All eigenvalues must be non-negative"
        assert np.all(np.diff(eigenvalues) >= -1e-12), "Eigenvalues must be sorted ascending"

    def test_floating_base_rigid_modes(self) -> None:
        """First 6 modes should have ω² ≈ 0 when K has 6 zero eigenvalues."""
        n = 29  # typical nv for floating-base G1
        M = _make_spd(n, seed=7)
        K = _make_spsd(n)  # first 6 diagonal entries are zero
        eigenvalues, _ = compute_modal_decomposition(M, K, n_modes=n)
        np.testing.assert_allclose(eigenvalues[:6], 0.0, atol=1e-6)

    def test_n_modes_respected(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """Returned mode_shapes must have exactly n_modes columns."""
        for n_modes in [1, 3, 5]:
            eigenvalues, mode_shapes = compute_modal_decomposition(
                small_mass_matrix, small_stiffness_matrix, n_modes=n_modes
            )
            assert eigenvalues.shape == (n_modes,), (
                f"eigenvalues shape mismatch for n_modes={n_modes}"
            )
            assert mode_shapes.shape == (5, n_modes), (
                f"mode_shapes shape mismatch for n_modes={n_modes}"
            )

    def test_n_modes_clamped_to_n(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """n_modes larger than the system size must be clamped to n."""
        eigenvalues, mode_shapes = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=100
        )
        assert eigenvalues.shape == (5,)
        assert mode_shapes.shape == (5, 5)

    def test_invalid_n_modes_raises(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """n_modes <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="n_modes"):
            compute_modal_decomposition(small_mass_matrix, small_stiffness_matrix, n_modes=0)

    def test_mismatched_shapes_raises(self) -> None:
        """K and M with different shapes must raise ValueError."""
        M = np.eye(4)
        K = np.eye(5)
        with pytest.raises(ValueError, match="same shape"):
            compute_modal_decomposition(M, K)

    def test_non_square_M_raises(self) -> None:
        """Non-square M must raise ValueError."""
        M = np.ones((3, 4))
        K = np.eye(4)
        with pytest.raises(ValueError, match="square"):
            compute_modal_decomposition(M, K)


class TestIsMOrthonormal:
    def test_is_m_orthonormal_true(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """Mode shapes from compute_modal_decomposition must be M-orthonormal."""
        _, mode_shapes = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=5
        )
        assert is_m_orthonormal(mode_shapes, small_mass_matrix, atol=1e-8)

    def test_is_m_orthonormal_false(
        self,
        small_mass_matrix: np.ndarray,
    ) -> None:
        """Random (un-orthonormalised) vectors must fail the M-orthonormality check."""
        rng = np.random.RandomState(99)
        mode_shapes = rng.randn(5, 5)  # not M-orthonormal in general
        # It would be astronomically unlikely for a random matrix to be M-orthonormal
        assert not is_m_orthonormal(mode_shapes, small_mass_matrix, atol=1e-6)

    def test_is_m_orthonormal_tight_tolerance(
        self,
        small_mass_matrix: np.ndarray,
        small_stiffness_matrix: np.ndarray,
    ) -> None:
        """A slightly perturbed basis must fail with a tight tolerance."""
        _, mode_shapes = compute_modal_decomposition(
            small_mass_matrix, small_stiffness_matrix, n_modes=5
        )
        # Perturb one column slightly
        perturbed = mode_shapes.copy()
        perturbed[:, 0] += 1e-5
        # Should fail at a very tight tolerance
        assert not is_m_orthonormal(perturbed, small_mass_matrix, atol=1e-12)


class TestPrecomputeG1Modes:
    def test_precompute_g1_modes_returns_dict(self, tmp_path: "pytest.TempPathFactory") -> None:
        """precompute_g1_modes must return a dict keyed by config name."""
        configs = [
            ("standing", np.zeros(23, dtype=np.float64)),
            ("crouched", np.full(23, 0.1, dtype=np.float64)),
        ]
        results = precompute_g1_modes(
            configs, n_modes=5, cache_dir=str(tmp_path / "g1_modes")
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {"standing", "crouched"}

    def test_precompute_g1_modes_output_shapes(self, tmp_path: "pytest.TempPathFactory") -> None:
        """Each result entry must contain 'eigenvalues', 'mode_shapes', 'q_ref'."""
        q_ref = np.zeros(23, dtype=np.float64)
        configs = [("test_config", q_ref)]
        results = precompute_g1_modes(
            configs, n_modes=4, cache_dir=str(tmp_path / "g1_modes_shapes")
        )
        entry = results["test_config"]
        assert "eigenvalues" in entry
        assert "mode_shapes" in entry
        assert "q_ref" in entry
        assert entry["eigenvalues"].shape == (4,)
        assert entry["mode_shapes"].shape[1] == 4
        np.testing.assert_array_equal(entry["q_ref"], q_ref)

    def test_precompute_g1_modes_writes_npz(self, tmp_path: "pytest.TempPathFactory") -> None:
        """Each config must produce a .npz file in cache_dir."""
        import pathlib

        cache_dir = str(tmp_path / "g1_modes_npz")
        configs = [("alpha", np.zeros(10, dtype=np.float64))]
        precompute_g1_modes(configs, n_modes=3, cache_dir=cache_dir)
        npz_path = pathlib.Path(cache_dir) / "alpha.npz"
        assert npz_path.exists(), f"Expected {npz_path} to exist"

    def test_precompute_g1_modes_eigenvalues_nonneg(
        self, tmp_path: "pytest.TempPathFactory"
    ) -> None:
        """All eigenvalues returned must be non-negative."""
        configs = [("nonneg_check", np.zeros(15, dtype=np.float64))]
        results = precompute_g1_modes(
            configs, n_modes=8, cache_dir=str(tmp_path / "nonneg")
        )
        eigenvalues = results["nonneg_check"]["eigenvalues"]
        assert np.all(eigenvalues >= 0.0), "Eigenvalues must be non-negative"
