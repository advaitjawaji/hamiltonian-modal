"""Symplectic integration interfaces.

Two time-reversible, symplectic integrators for Hamiltonian systems
H(q, p) = T(p) + V(q):

* :func:`leapfrog_step` — Störmer–Verlet / leapfrog (velocity form).
* :func:`stormer_verlet_step` — position-form Störmer–Verlet (equivalent
  to leapfrog but expressed in (q, p) instead of (q, v)).

Both conserve a *shadow Hamiltonian* to O(dt²) and are time-reversible,
making them far superior to Euler integration for long rollouts.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "leapfrog_step",
    "stormer_verlet_step",
    "integrate_trajectory",
]

GradFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def leapfrog_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    grad_H_q: GradFn,
    grad_H_p: GradFn,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perform one leapfrog (velocity-Störmer–Verlet) integration step.

    Update rule:

    .. math::

        p_{1/2} &= p_0 - \\tfrac{dt}{2}\\,\\nabla_q H(q_0, p_0) \\\\
        q_1     &= q_0 + dt\\,\\nabla_p H(q_0, p_{1/2}) \\\\
        p_1     &= p_{1/2} - \\tfrac{dt}{2}\\,\\nabla_q H(q_1, p_{1/2})

    Parameters
    ----------
    q:
        Generalised coordinates, shape ``(n,)``.
    p:
        Generalised momenta, shape ``(n,)``.
    grad_H_q:
        Callable ``q → ∂H/∂q``.
    grad_H_p:
        Callable ``p → ∂H/∂p = M⁻¹ p`` (velocity from momentum).
    dt:
        Integration timestep in seconds.

    Returns
    -------
    tuple[NDArray, NDArray]
        Updated ``(q₁, p₁)``.
    """
    p_half = p - 0.5 * dt * grad_H_q(q)
    q_new = q + dt * grad_H_p(p_half)
    p_new = p_half - 0.5 * dt * grad_H_q(q_new)
    return q_new, p_new


def stormer_verlet_step(
    q: NDArray[np.float64],
    p: NDArray[np.float64],
    grad_V_q: GradFn,
    M_inv: NDArray[np.float64],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Perform one position-form Störmer–Verlet step.

    Suitable when the Hamiltonian is separable: H(q,p) = ½ pᵀ M⁻¹ p + V(q).

    Parameters
    ----------
    q:
        Generalised coordinates, shape ``(n,)``.
    p:
        Generalised momenta, shape ``(n,)``.
    grad_V_q:
        Callable ``q → ∂V/∂q`` (conservative force).
    M_inv:
        Inverse mass matrix, shape ``(n, n)``.
    dt:
        Integration timestep in seconds.

    Returns
    -------
    tuple[NDArray, NDArray]
        Updated ``(q₁, p₁)``.
    """
    p_half = p - 0.5 * dt * grad_V_q(q)
    q_new = q + dt * (M_inv @ p_half)
    p_new = p_half - 0.5 * dt * grad_V_q(q_new)
    return q_new, p_new


def integrate_trajectory(
    q0: NDArray[np.float64],
    p0: NDArray[np.float64],
    grad_H_q: GradFn,
    grad_H_p: GradFn,
    dt: float,
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate a Hamiltonian trajectory using leapfrog.

    Parameters
    ----------
    q0:
        Initial generalised coordinates, shape ``(n,)``.
    p0:
        Initial generalised momenta, shape ``(n,)``.
    grad_H_q:
        Callable ``q → ∂H/∂q``.
    grad_H_p:
        Callable ``p → ∂H/∂p``.
    dt:
        Integration timestep in seconds.
    n_steps:
        Number of leapfrog steps.

    Returns
    -------
    tuple[NDArray, NDArray]
        Arrays ``qs`` of shape ``(n_steps+1, n)`` and ``ps`` of shape
        ``(n_steps+1, n)`` including the initial conditions.
    """
    n = q0.shape[0]
    qs = np.empty((n_steps + 1, n), dtype=np.float64)
    ps = np.empty((n_steps + 1, n), dtype=np.float64)
    qs[0] = q0
    ps[0] = p0
    q, p = q0.copy(), p0.copy()
    for i in range(n_steps):
        q, p = leapfrog_step(q, p, grad_H_q, grad_H_p, dt)
        qs[i + 1] = q
        ps[i + 1] = p
    return qs, ps
