"""Unit tests for hamiltonian_modal.utils.pinocchio_wrapper.

All tests in this module are skipped automatically if ``pinocchio`` is not
installed (via ``pytest.importorskip`` at module level).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip every test in this module if pinocchio is not available
pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")


from hamiltonian_modal.utils.pinocchio_wrapper import load_g1_model, mass_matrix_at


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def g1_model_and_data():
    """Load the G1 model once per test module.

    Skips if the URDF cannot be found (Genesis not installed and no URDF
    path available).
    """
    try:
        model, data = load_g1_model()
    except FileNotFoundError as exc:
        pytest.skip(f"G1 URDF not available: {exc}")
    return model, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadG1Model:
    def test_returns_model_and_data(self, g1_model_and_data) -> None:
        """load_g1_model must return a (model, data) tuple."""
        model, data = g1_model_and_data
        # Pinocchio model and data have nq and nv attributes
        assert hasattr(model, "nq")
        assert hasattr(model, "nv")
        assert hasattr(data, "M")

    def test_model_has_positive_dofs(self, g1_model_and_data) -> None:
        """G1 model must have positive nq and nv."""
        model, _ = g1_model_and_data
        assert model.nq > 0
        assert model.nv > 0

    def test_model_has_bodies(self, g1_model_and_data) -> None:
        """G1 model must have multiple rigid bodies."""
        model, _ = g1_model_and_data
        assert model.nbodies > 1

    def test_invalid_urdf_path_raises(self) -> None:
        """Providing a non-existent URDF path must raise an error."""
        with pytest.raises(Exception):
            # pin.buildModelFromUrdf raises RuntimeError for missing files
            load_g1_model(urdf_path="/nonexistent/path/to/g1.urdf")


class TestMassMatrixAt:
    def test_mass_matrix_pd(self, g1_model_and_data) -> None:
        """M(q) must be symmetric positive definite at the zero configuration."""
        model, data = g1_model_and_data

        # Construct a valid neutral configuration
        q = pin.neutral(model)  # type: ignore[attr-defined]
        M = mass_matrix_at(model, data, q)

        # Check shape
        assert M.shape == (model.nv, model.nv)

        # Check symmetry
        np.testing.assert_allclose(M, M.T, atol=1e-10)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0), (
            f"Mass matrix is not positive definite. Min eigenvalue: {eigenvalues.min()}"
        )

    def test_mass_matrix_symmetry(self, g1_model_and_data) -> None:
        """M(q) must be symmetric for an arbitrary configuration."""
        model, data = g1_model_and_data
        rng = np.random.RandomState(123)
        q = pin.randomConfiguration(model, rng)  # type: ignore[attr-defined]
        M = mass_matrix_at(model, data, q)
        np.testing.assert_allclose(M, M.T, atol=1e-10)

    def test_mass_matrix_shape(self, g1_model_and_data) -> None:
        """M(q) must have shape (nv, nv)."""
        model, data = g1_model_and_data
        q = pin.neutral(model)  # type: ignore[attr-defined]
        M = mass_matrix_at(model, data, q)
        assert M.shape == (model.nv, model.nv)

    def test_mass_matrix_varies_with_config(self, g1_model_and_data) -> None:
        """M(q) must differ for different configurations (not a constant matrix)."""
        model, data = g1_model_and_data
        q1 = pin.neutral(model)  # type: ignore[attr-defined]
        rng = np.random.RandomState(7)
        q2 = pin.randomConfiguration(model, rng)  # type: ignore[attr-defined]
        M1 = mass_matrix_at(model, data, q1)
        M2 = mass_matrix_at(model, data, q2)
        # For a robot with multiple links, the mass matrix is configuration-dependent
        assert not np.allclose(M1, M2, atol=1e-3), (
            "Mass matrix should vary with configuration for a multi-link robot"
        )

    def test_mass_matrix_dtype(self, g1_model_and_data) -> None:
        """Returned M must have float64 dtype."""
        model, data = g1_model_and_data
        q = pin.neutral(model)  # type: ignore[attr-defined]
        M = mass_matrix_at(model, data, q)
        assert M.dtype == np.float64
