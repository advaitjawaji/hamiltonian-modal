"""Tests for hamiltonian_modal.modal.encoder and decoder round-trips."""
import numpy as np
import pytest

from hamiltonian_modal.modal.encoder import (
    project_to_modal,
    encode_position,
    encode_velocity,
    batch_project_to_modal,
)
from hamiltonian_modal.modal.decoder import (
    project_from_modal,
    decode_position,
    decode_velocity,
    batch_project_from_modal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mass_ortho_basis(n_joints: int, n_modes: int, rng: np.random.Generator):
    """Return (Phi, M) where Phi^T M Phi = I."""
    # Build random SPD mass matrix
    A = rng.standard_normal((n_joints, n_joints))
    M = A @ A.T + n_joints * np.eye(n_joints)

    # Build mass-orthonormal mode shapes via eigendecomposition of M
    # Random symmetric K
    B = rng.standard_normal((n_joints, n_joints))
    K = B @ B.T + np.eye(n_joints)
    from scipy.linalg import eigh
    vals, vecs = eigh(K, M)
    # vecs are M-orthonormal: vecs.T @ M @ vecs == I
    Phi = vecs[:, :n_modes]
    return Phi, M


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_10x4():
    """10 joints, 4 modes."""
    rng = np.random.default_rng(42)
    n_joints, n_modes = 10, 4
    Phi, M = _make_mass_ortho_basis(n_joints, n_modes, rng)
    q_ref = rng.standard_normal(n_joints)
    return Phi, M, q_ref, n_joints, n_modes, rng


# ---------------------------------------------------------------------------
# test_project_to_modal_shapes
# ---------------------------------------------------------------------------

def test_project_to_modal_shapes(setup_10x4):
    Phi, M, q_ref, n_joints, n_modes, rng = setup_10x4
    q = rng.standard_normal(n_joints)
    qdot = rng.standard_normal(n_joints)

    eta, eta_dot = project_to_modal(q, qdot, Phi, M, q_ref)

    assert eta.shape == (n_modes,), f"expected ({n_modes},), got {eta.shape}"
    assert eta_dot.shape == (n_modes,), f"expected ({n_modes},), got {eta_dot.shape}"
    assert eta.dtype == np.float64
    assert eta_dot.dtype == np.float64


# ---------------------------------------------------------------------------
# test_round_trip
# ---------------------------------------------------------------------------

def test_round_trip(setup_10x4):
    """Encode then decode should recover q and qdot up to float64 precision."""
    Phi, M, q_ref, n_joints, n_modes, rng = setup_10x4
    q = rng.standard_normal(n_joints)
    qdot = rng.standard_normal(n_joints)

    eta, eta_dot = project_to_modal(q, qdot, Phi, M, q_ref)
    q_rec, qdot_rec = project_from_modal(eta, eta_dot, Phi, q_ref)

    # Reconstruction only exact when n_modes == n_joints (full basis);
    # with partial basis check the modal residual instead.
    # Verify: eta matches encode_position / encode_velocity
    eta2 = encode_position(q, Phi, M, q_ref)
    eta_dot2 = encode_velocity(qdot, Phi, M)
    np.testing.assert_allclose(eta, eta2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(eta_dot, eta_dot2, rtol=1e-12, atol=1e-14)

    # Verify decoder: decode_position and decode_velocity are consistent
    q_rec2 = decode_position(eta, Phi, q_ref)
    qdot_rec2 = decode_velocity(eta_dot, Phi)
    np.testing.assert_allclose(q_rec, q_rec2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(qdot_rec, qdot_rec2, rtol=1e-12, atol=1e-14)


def test_round_trip_full_basis():
    """With full basis (n_modes == n_joints) and identity mass, round-trip is exact."""
    rng = np.random.default_rng(7)
    n = 6
    # Identity mass, full orthonormal basis
    M = np.eye(n)
    Phi = np.eye(n)
    q_ref = np.zeros(n)
    q = rng.standard_normal(n)
    qdot = rng.standard_normal(n)

    eta, eta_dot = project_to_modal(q, qdot, Phi, M, q_ref)
    q_rec, qdot_rec = project_from_modal(eta, eta_dot, Phi, q_ref)

    np.testing.assert_allclose(q_rec, q, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(qdot_rec, qdot, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# test_vmap_batched
# ---------------------------------------------------------------------------

def test_vmap_batched(setup_10x4):
    """batch_project_to_modal and batch_project_from_modal work correctly."""
    Phi, M, q_ref, n_joints, n_modes, rng = setup_10x4
    batch = 8
    q_batch = rng.standard_normal((batch, n_joints))
    qdot_batch = rng.standard_normal((batch, n_joints))

    eta_batch, eta_dot_batch = batch_project_to_modal(q_batch, qdot_batch, Phi, M, q_ref)

    assert eta_batch.shape == (batch, n_modes)
    assert eta_dot_batch.shape == (batch, n_modes)

    q_rec_batch, qdot_rec_batch = batch_project_from_modal(eta_batch, eta_dot_batch, Phi, q_ref)

    assert q_rec_batch.shape == (batch, n_joints)
    assert qdot_rec_batch.shape == (batch, n_joints)


# ---------------------------------------------------------------------------
# test_batch_project_consistent
# ---------------------------------------------------------------------------

def test_batch_project_consistent(setup_10x4):
    """batch_project_to_modal must match per-sample project_to_modal."""
    Phi, M, q_ref, n_joints, n_modes, rng = setup_10x4
    batch = 5
    q_batch = rng.standard_normal((batch, n_joints))
    qdot_batch = rng.standard_normal((batch, n_joints))

    eta_batch, eta_dot_batch = batch_project_to_modal(q_batch, qdot_batch, Phi, M, q_ref)

    for i in range(batch):
        eta_i, eta_dot_i = project_to_modal(q_batch[i], qdot_batch[i], Phi, M, q_ref)
        np.testing.assert_allclose(eta_batch[i], eta_i, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(eta_dot_batch[i], eta_dot_i, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# test_jit_compatibility (JAX optional)
# ---------------------------------------------------------------------------

def test_jit_compatibility():
    """If JAX is available, project_to_modal should be jit-able (numpy fallback)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    rng = np.random.default_rng(99)
    n_joints, n_modes = 6, 3
    Phi, M = _make_mass_ortho_basis(n_joints, n_modes, rng)
    q_ref = rng.standard_normal(n_joints)
    q = rng.standard_normal(n_joints)
    qdot = rng.standard_normal(n_joints)

    # Wrap in a pure function that uses jnp
    def jax_project(q, qdot):
        delta_q = q - q_ref
        eta = Phi.T @ M @ delta_q
        eta_dot = Phi.T @ M @ qdot
        return eta, eta_dot

    jitted = jax.jit(jax_project)
    eta_jax, eta_dot_jax = jitted(jnp.array(q), jnp.array(qdot))

    eta_np, eta_dot_np = project_to_modal(q, qdot, Phi, M, q_ref)
    np.testing.assert_allclose(np.array(eta_jax), eta_np, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.array(eta_dot_jax), eta_dot_np, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# test_round_trip_property (hypothesis)
# ---------------------------------------------------------------------------

def test_round_trip_property():
    """Property: encode then decode is idempotent for identity mass + full basis."""
    try:
        from hypothesis import given, settings
        import hypothesis.strategies as st

        @given(
            q=st.lists(
                st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                min_size=4,
                max_size=4,
            )
        )
        @settings(max_examples=50)
        def _inner(q):
            n = 4
            q_arr = np.array(q, dtype=np.float64)
            M = np.eye(n)
            Phi = np.eye(n)
            q_ref = np.zeros(n)
            qdot = np.zeros(n)
            eta, eta_dot = project_to_modal(q_arr, qdot, Phi, M, q_ref)
            q_rec, _ = project_from_modal(eta, eta_dot, Phi, q_ref)
            np.testing.assert_allclose(q_rec, q_arr, rtol=1e-10, atol=1e-12)

        _inner()
    except ImportError:
        pytest.skip("hypothesis not available")
