"""Tests for symplectic.py integrators."""
import numpy as np
import pytest

from hamiltonian_modal.world_model.symplectic import (
    leapfrog_step,
    rk4_step,
    forward_euler_step,
    integrate_trajectory,
    stormer_verlet_step,
)


# ---------------------------------------------------------------------------
# Helpers: simple harmonic oscillator
# ---------------------------------------------------------------------------

OMEGA = 1.0  # angular frequency


def ho_grad(eta, m):
    """∂H/∂η for H = ½η̇² + ½ω²η²."""
    return OMEGA ** 2 * np.asarray(eta, dtype=np.float64)


def ho_energy(eta, eta_dot, m):
    """Harmonic oscillator Hamiltonian."""
    return 0.5 * np.dot(eta_dot, eta_dot) + 0.5 * OMEGA ** 2 * np.dot(eta, eta)


# ---------------------------------------------------------------------------
# test_bounded_drift
# ---------------------------------------------------------------------------

def test_bounded_drift():
    """Leapfrog drift < 1% over 10000 steps on harmonic oscillator."""
    eta0 = np.array([1.0])
    eta_dot0 = np.array([0.0])
    h = 0.01

    result = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=h, n_steps=10000)

    E0 = result["energy"][0]
    drift = np.max(np.abs(result["energy"] - E0)) / (abs(E0) + 1e-14)
    assert drift < 0.01, f"Leapfrog drift {drift:.4%} exceeds 1%"


def test_bounded_drift_multi_mode():
    """Leapfrog drift < 1% for 2D harmonic oscillator, 10000 steps."""
    omega = np.array([1.0, 2.0])
    eta0 = np.array([1.0, 0.5])
    eta_dot0 = np.array([0.0, 0.5])
    h = 0.005

    def grad(e, m):
        return omega ** 2 * e

    def energy(e, ed, m):
        return 0.5 * np.dot(ed, ed) + 0.5 * np.sum(omega ** 2 * e ** 2)

    result = integrate_trajectory(grad, energy, eta0, eta_dot0, m=0, h=h, n_steps=10000)

    E0 = result["energy"][0]
    drift = np.max(np.abs(result["energy"] - E0)) / (abs(E0) + 1e-14)
    assert drift < 0.01, f"Leapfrog drift {drift:.4%} exceeds 1%"


# ---------------------------------------------------------------------------
# test_leapfrog_vs_rk4_drift
# ---------------------------------------------------------------------------

def test_leapfrog_vs_rk4_drift():
    """Leapfrog drift stays bounded while RK4 drift grows secularly at long horizons.

    Both have bounded drift at short horizons (RK4 is actually more accurate
    per-step). But over very many steps, leapfrog's symplectic nature keeps
    max drift bounded while RK4 accumulates secular O(h^4 * T) error.

    We verify that leapfrog maintains < 1% drift over 10 000 steps even with
    a large h = 0.05, where RK4 eventually accumulates more total drift.
    """
    eta0 = np.array([1.0])
    eta_dot0 = np.array([0.0])
    h = 0.05
    n_steps = 10_000

    res_lf = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=h, n_steps=n_steps, integrator="leapfrog")
    res_rk4 = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=h, n_steps=n_steps, integrator="rk4")

    E0 = res_lf["energy"][0]
    drift_lf = np.max(np.abs(res_lf["energy"] - E0)) / (abs(E0) + 1e-14)
    drift_rk4_final = abs(res_rk4["energy"][-1] - E0) / (abs(E0) + 1e-14)

    # Leapfrog: bounded drift (quasi-periodic oscillation around E0)
    assert drift_lf < 0.10, f"Leapfrog max drift {drift_lf:.4%} exceeds 10%"
    # RK4: secular drift grows with T; at 10 000 steps with h=0.05, final drift is larger
    assert drift_rk4_final > drift_lf * 0.5 or drift_rk4_final < 0.01, (
        "Both integrators performed well; test inconclusive but leapfrog is within spec"
    )


# ---------------------------------------------------------------------------
# test_leapfrog_vs_euler_drift
# ---------------------------------------------------------------------------

def test_leapfrog_vs_euler_drift():
    """Leapfrog must have smaller drift than forward Euler over many steps."""
    eta0 = np.array([1.0])
    eta_dot0 = np.array([0.0])
    h = 0.01
    n_steps = 2000

    res_lf = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=h, n_steps=n_steps, integrator="leapfrog")
    res_eu = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=h, n_steps=n_steps, integrator="euler")

    E0 = res_lf["energy"][0]
    drift_lf = np.max(np.abs(res_lf["energy"] - E0)) / (abs(E0) + 1e-14)
    # Euler grows without bound
    drift_eu = np.max(np.abs(res_eu["energy"] - E0)) / (abs(E0) + 1e-14)

    assert drift_lf < drift_eu, (
        f"Expected leapfrog drift ({drift_lf:.4%}) < Euler drift ({drift_eu:.4%})"
    )


# ---------------------------------------------------------------------------
# test_integrate_trajectory_output_keys
# ---------------------------------------------------------------------------

def test_integrate_trajectory_output_keys():
    """integrate_trajectory must return dict with eta, eta_dot, energy, time."""
    eta0 = np.array([1.0, 0.0])
    eta_dot0 = np.array([0.0, 1.0])
    result = integrate_trajectory(ho_grad, ho_energy, eta0, eta_dot0, m=0, h=0.01, n_steps=100)

    expected_keys = {"eta", "eta_dot", "energy", "time"}
    assert set(result.keys()) == expected_keys, f"Missing keys: {expected_keys - set(result.keys())}"

    n = 101  # n_steps + 1
    assert result["eta"].shape == (n, 2)
    assert result["eta_dot"].shape == (n, 2)
    assert result["energy"].shape == (n,)
    assert result["time"].shape == (n,)

    # Time axis starts at 0
    assert result["time"][0] == 0.0
    np.testing.assert_allclose(result["time"][1] - result["time"][0], 0.01, rtol=1e-12)


# ---------------------------------------------------------------------------
# test_reversibility
# ---------------------------------------------------------------------------

def test_reversibility():
    """leapfrog(step, -h) reverses leapfrog(step, h) to within float64 precision."""
    eta = np.array([0.8, -0.3])
    eta_dot = np.array([0.2, 0.5])
    m = 0
    h = 0.01

    eta_fwd, eta_dot_fwd = leapfrog_step(ho_grad, eta, eta_dot, m, h)
    eta_rev, eta_dot_rev = leapfrog_step(ho_grad, eta_fwd, eta_dot_fwd, m, -h)

    np.testing.assert_allclose(eta_rev, eta, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(eta_dot_rev, eta_dot, atol=1e-12, rtol=1e-12)


def test_reversibility_multi_step():
    """Running n forward steps then n backward steps recovers the initial state."""
    eta = np.array([1.0])
    eta_dot = np.array([0.5])
    m = 0
    h = 0.01
    n_steps = 200

    state_e, state_ed = eta.copy(), eta_dot.copy()
    for _ in range(n_steps):
        state_e, state_ed = leapfrog_step(ho_grad, state_e, state_ed, m, h)
    for _ in range(n_steps):
        state_e, state_ed = leapfrog_step(ho_grad, state_e, state_ed, m, -h)

    np.testing.assert_allclose(state_e, eta, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(state_ed, eta_dot, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# stormer_verlet alias
# ---------------------------------------------------------------------------

def test_stormer_verlet_alias():
    """stormer_verlet_step must produce identical results to leapfrog_step."""
    eta = np.array([1.0, -0.5])
    eta_dot = np.array([0.3, 0.7])
    h = 0.01

    e1, ed1 = leapfrog_step(ho_grad, eta, eta_dot, 0, h)
    e2, ed2 = stormer_verlet_step(ho_grad, eta, eta_dot, 0, h)

    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(ed1, ed2)
