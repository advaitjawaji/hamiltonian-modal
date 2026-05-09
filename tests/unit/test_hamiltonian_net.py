"""Tests for HamiltonianNet."""
import numpy as np
import pytest

from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_net():
    """Small HamiltonianNet for fast tests."""
    return HamiltonianNet(
        n_modes=4,
        n_contact_modes=3,
        v_hidden=[32, 32],
        contact_hidden=[16, 16],
        seed=0,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(123)


# ---------------------------------------------------------------------------
# test_hamiltonian_net_init
# ---------------------------------------------------------------------------

def test_hamiltonian_net_init():
    """HamiltonianNet initializes with correct attributes."""
    net = HamiltonianNet(n_modes=6, n_contact_modes=5, seed=42)
    assert net.n_modes == 6
    assert net.n_contact_modes == 5
    assert net.V_net is not None
    assert net.contact_net is not None


def test_hamiltonian_net_init_custom_hidden():
    net = HamiltonianNet(
        n_modes=8,
        n_contact_modes=4,
        v_hidden=[64, 64],
        contact_hidden=[32],
        seed=1,
    )
    assert net.n_modes == 8
    # Verify MLP layer count: input + 2 hidden + output = 3 weight matrices
    assert len(net.V_net.weights) == 3  # [8+4->64, 64->64, 64->1]
    assert len(net.contact_net.weights) == 2  # [8+4->32, 32->1]


# ---------------------------------------------------------------------------
# test_kinetic_energy_positive
# ---------------------------------------------------------------------------

def test_kinetic_energy_positive(small_net, rng):
    """T(η̇) = ½‖η̇‖² ≥ 0 always."""
    for _ in range(20):
        eta_dot = rng.standard_normal(small_net.n_modes)
        T = small_net.kinetic(eta_dot)
        assert T >= 0.0

    # Zero velocity → zero kinetic energy
    T_zero = small_net.kinetic(np.zeros(small_net.n_modes))
    assert T_zero == 0.0


def test_kinetic_energy_formula(small_net, rng):
    """T(η̇) must equal 0.5 * sum(η̇²)."""
    eta_dot = rng.standard_normal(small_net.n_modes)
    expected = 0.5 * float(np.dot(eta_dot, eta_dot))
    assert abs(small_net.kinetic(eta_dot) - expected) < 1e-14


# ---------------------------------------------------------------------------
# test_hamilton_equations
# ---------------------------------------------------------------------------

def test_hamilton_equations(small_net, rng):
    """∂H/∂η̇ = η̇ for T = ½‖η̇‖² (analytical gradient)."""
    for _ in range(10):
        eta = rng.standard_normal(small_net.n_modes)
        eta_dot = rng.standard_normal(small_net.n_modes)
        m = rng.integers(0, small_net.n_contact_modes)

        grad_eta_dot = small_net.grad_H_eta_dot(eta, eta_dot, m)
        np.testing.assert_allclose(grad_eta_dot, eta_dot, rtol=1e-12, atol=1e-14)


def test_grad_H_eta_finite_diff_consistency(small_net, rng):
    """grad_H_eta via central FD must match a manual FD computation."""
    eta = rng.standard_normal(small_net.n_modes)
    eta_dot = rng.standard_normal(small_net.n_modes)
    m = 1

    grad = small_net.grad_H_eta(eta, eta_dot, m, eps=1e-6)

    # Manual FD reference with same eps
    eps = 1e-6
    grad_ref = np.zeros_like(eta)
    for i in range(len(eta)):
        ep = eta.copy(); ep[i] += eps
        em = eta.copy(); em[i] -= eps
        grad_ref[i] = (small_net(ep, eta_dot, m) - small_net(em, eta_dot, m)) / (2 * eps)

    np.testing.assert_allclose(grad, grad_ref, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# test_hamilton_step_shape
# ---------------------------------------------------------------------------

def test_hamilton_step_shape(small_net, rng):
    """hamilton_step returns (eta_new, eta_dot_new) with correct shapes."""
    eta = rng.standard_normal(small_net.n_modes)
    eta_dot = rng.standard_normal(small_net.n_modes)
    m = 0

    eta_new, eta_dot_new = small_net.hamilton_step(eta, eta_dot, m, h=0.01)

    assert eta_new.shape == (small_net.n_modes,)
    assert eta_dot_new.shape == (small_net.n_modes,)
    assert eta_new.dtype == np.float64
    assert eta_dot_new.dtype == np.float64


# ---------------------------------------------------------------------------
# test_contact_correction_shape
# ---------------------------------------------------------------------------

def test_contact_correction_shape(small_net, rng):
    """contact_correction returns a scalar."""
    eta = rng.standard_normal(small_net.n_modes)
    for m in range(small_net.n_contact_modes):
        cc = small_net.contact_correction(eta, m)
        assert isinstance(cc, float), f"expected float, got {type(cc)}"


def test_potential_shape(small_net, rng):
    """potential returns a scalar."""
    eta = rng.standard_normal(small_net.n_modes)
    for m in range(small_net.n_contact_modes):
        V = small_net.potential(eta, m)
        assert isinstance(V, float)


# ---------------------------------------------------------------------------
# test_energy_conservation_free_vibration
# ---------------------------------------------------------------------------

def test_energy_conservation_free_vibration():
    """Leapfrog energy drift < 5% over 1000 steps on a simple harmonic oscillator.

    We replace V_theta with the true quadratic V = ½ω²η², which means the
    Hamiltonian net's gradient must agree with the analytical one.  We do this
    by using the symplectic integrator directly with the analytical gradient.
    """
    from hamiltonian_modal.world_model.symplectic import integrate_trajectory

    omega = 2.0  # rad/s
    n_modes = 2
    eta0 = np.array([1.0, 0.5])
    eta_dot0 = np.array([0.0, 0.0])

    def H_grad(e, m):
        return omega ** 2 * e  # ∂V/∂η for V = ½ω²‖η‖²

    def H_energy(e, ed, m):
        return 0.5 * np.dot(ed, ed) + 0.5 * omega ** 2 * np.dot(e, e)

    result = integrate_trajectory(H_grad, H_energy, eta0, eta_dot0, m=0, h=0.01, n_steps=1000)

    E0 = result["energy"][0]
    E_max_drift = np.max(np.abs(result["energy"] - E0)) / (abs(E0) + 1e-12)
    assert E_max_drift < 0.05, f"Energy drift {E_max_drift:.4%} exceeds 5%"


def test_energy_conservation_hamiltonian_net_leapfrog():
    """HamiltonianNet.hamilton_step energy drift < 5% for 1000 steps (small net)."""
    net = HamiltonianNet(
        n_modes=2,
        n_contact_modes=2,
        v_hidden=[16, 16],
        contact_hidden=[8],
        seed=7,
    )
    rng = np.random.default_rng(0)
    eta = rng.standard_normal(2)
    eta_dot = rng.standard_normal(2)
    m = 0

    H0 = net(eta, eta_dot, m)
    energies = [H0]

    for _ in range(1000):
        eta, eta_dot = net.hamilton_step(eta, eta_dot, m, h=0.005)
        energies.append(net(eta, eta_dot, m))

    E0 = energies[0]
    E_arr = np.array(energies)
    drift = np.max(np.abs(E_arr - E0)) / (abs(E0) + 1e-12)
    # Leapfrog drift bounded by O(h²) per step
    assert drift < 0.05, f"Energy drift {drift:.4%} exceeds 5%"
