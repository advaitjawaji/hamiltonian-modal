"""Stiffness-model interfaces for Unitree G1.

Two construction strategies are provided:

* :func:`diagonal_stiffness` — builds a diagonal K from per-DoF joint-spring
  values.  This is the simplest approximation and sufficient for a
  linearisation around a static equilibrium.
* :func:`stiffness_from_gravity_hessian` — estimates K numerically as the
  finite-difference Hessian of the gravitational potential energy (i.e. the
  Jacobian of the generalised gravity vector g(q)) evaluated at an equilibrium
  configuration ``q_eq``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "diagonal_stiffness",
    "stiffness_from_gravity_hessian",
]


def diagonal_stiffness(
    n_dof: int,
    k_values: ArrayLike | float = 1.0,
) -> NDArray[np.float64]:
    """Build a diagonal stiffness matrix K from per-DoF spring constants.

    Parameters
    ----------
    n_dof:
        Number of degrees of freedom.
    k_values:
        Scalar or array-like of length *n_dof* with non-negative stiffness
        values (N m / rad for revolute joints).  A scalar is broadcast to all
        DoFs.

    Returns
    -------
    NDArray[np.float64]
        Symmetric positive semi-definite diagonal matrix of shape
        ``(n_dof, n_dof)``.

    Raises
    ------
    ValueError
        If *k_values* cannot be broadcast to shape ``(n_dof,)`` or contains
        negative entries.
    """
    k_arr = np.asarray(k_values, dtype=np.float64)
    if k_arr.ndim == 0:
        k_arr = np.full(n_dof, float(k_arr))
    else:
        k_arr = k_arr.reshape(-1)
        if k_arr.shape[0] != n_dof:
            raise ValueError(
                f"k_values length {k_arr.shape[0]} does not match n_dof={n_dof}."
            )
    if np.any(k_arr < 0):
        raise ValueError("Stiffness values must be non-negative.")
    return np.diag(k_arr)


def stiffness_from_gravity_hessian(
    gravity_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    q_eq: ArrayLike,
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Estimate the stiffness matrix K as the finite-difference Jacobian of the
    generalised gravity force vector g(q) evaluated at equilibrium *q_eq*.

    At a static equilibrium the potential-energy Hessian ∂²V/∂q² equals the
    Jacobian of g(q) = ∂V/∂q (up to sign convention).  Central differences
    with step *eps* are used.

    Parameters
    ----------
    gravity_fn:
        Callable ``q → g(q)`` that returns the generalised gravity force
        vector of shape ``(n_dof,)`` given a joint configuration ``q`` of
        the same length.  With Pinocchio this is
        ``pinocchio.computeGeneralizedGravity(model, data, q)``.
    q_eq:
        Equilibrium configuration, shape ``(n_dof,)``.
    eps:
        Finite-difference step size (radians).

    Returns
    -------
    NDArray[np.float64]
        Symmetric approximate stiffness matrix of shape ``(n_dof, n_dof)``.
    """
    q0 = np.asarray(q_eq, dtype=np.float64).reshape(-1)
    n_dof = q0.shape[0]
    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    for i in range(n_dof):
        q_plus = q0.copy()
        q_plus[i] += eps
        q_minus = q0.copy()
        q_minus[i] -= eps
        K[:, i] = (gravity_fn(q_plus) - gravity_fn(q_minus)) / (2.0 * eps)
    # Symmetrise to remove finite-difference asymmetry
    return 0.5 * (K + K.T)
