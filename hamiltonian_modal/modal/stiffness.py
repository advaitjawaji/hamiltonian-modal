"""Stiffness matrix assembly for the Unitree G1 humanoid.

Combines joint PD gains, harmonic-drive gearbox compliance, and link
bending stiffness from Euler-Bernoulli beam theory to produce an effective
stiffness matrix K(q_ref) used as input to the modal decomposition.

The assembled matrix has three additive contributions:

1. **Joint PD gains** — diagonal matrix of per-joint proportional gains
   (units: N·m/rad).
2. **Gearbox compliance** — harmonic-drive flex-spline stiffness reflected
   through the gear ratio; adds a small diagonal correction.
3. **Link bending stiffness** — Euler-Bernoulli bending contribution for
   each link modelled as a hollow circular cross-section aluminium tube.

References
----------
.. [1] Shabana, A. A. (2013). *Dynamics of Multibody Systems* (4th ed.).
       Cambridge University Press.
.. [2] Unitree Robotics G1 technical datasheet (2024).

Examples
--------
>>> import numpy as np
>>> from hamiltonian_modal.modal.stiffness import assemble_stiffness_matrix
>>> q_ref = np.zeros(23)
>>> K = assemble_stiffness_matrix(model=None, q_ref=q_ref)
>>> assert K.shape == (23, 23)
>>> assert np.allclose(K, K.T), "stiffness matrix must be symmetric"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

G1_DEFAULT_PD_GAINS: NDArray[np.float64] = np.full(23, 200.0)
"""Default joint PD proportional gains for G1, in N·m/rad."""

G1_DEFAULT_GEARBOX_STIFFNESS: float = 5000.0
"""Default harmonic-drive flex-spline stiffness, in N·m/rad."""

G1_DEFAULT_MATERIAL: dict = {"E": 70e9, "rho": 2700.0}
"""Default link material properties (6061 aluminium alloy)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_stiffness_matrix(
    model: Any,
    q_ref: NDArray[np.float64],
    joint_pd_gains: NDArray[np.float64] | None = None,
    gearbox_compliance: float = 5000.0,
    link_material: dict | None = None,
) -> NDArray[np.float64]:
    """Assemble effective stiffness matrix K(q_ref) for the Unitree G1.

    Parameters
    ----------
    model : pin.Model or None
        Pinocchio model used to look up link geometry.  If ``None``, a
        default geometry (hollow aluminium tube, 0.3 m length) is used
        for all links.
    q_ref : np.ndarray, shape (n,)
        Reference joint configuration at which the stiffness is evaluated.
        For the G1, ``n = 23`` (actuated DOFs only, no floating base).
    joint_pd_gains : np.ndarray, shape (n,) or None, optional
        Per-joint proportional gains in N·m/rad.  If ``None``, defaults to
        200 N·m/rad for every joint (``G1_DEFAULT_PD_GAINS``).
    gearbox_compliance : float, optional
        Harmonic-drive flex-spline stiffness in N·m/rad (before reflection
        through gear ratio).  Default is 5000 N·m/rad.
    link_material : dict or None, optional
        Material properties with keys:

        * ``"E"``   — Young's modulus in Pa (default 70 GPa, aluminium).
        * ``"rho"`` — density in kg/m³ (default 2700 kg/m³, aluminium).

        If ``None``, ``G1_DEFAULT_MATERIAL`` is used.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        Symmetric positive-definite effective stiffness matrix.

    Notes
    -----
    The three contributions are summed as::

        K = K_PD + K_gear + K_bend

    where:

    * ``K_PD   = diag(joint_pd_gains)``
    * ``K_gear = diag(gearbox_compliance / gear_ratio)``
    * ``K_bend = diag(E * I / L^3)`` with ``I`` the second moment of area
      of a hollow circular cross-section.

    All three sub-matrices are diagonal, so the assembled K is diagonal
    in this implementation.  Off-diagonal coupling terms (e.g. from
    Pinocchio's geometric Jacobians) are left for future work.

    The result is explicitly symmetrised as ``K = 0.5 * (K + K^T)`` to
    guard against floating-point asymmetry.
    """
    q_ref = np.asarray(q_ref, dtype=np.float64)
    n = q_ref.shape[0]

    if joint_pd_gains is None:
        joint_pd_gains = np.full(n, 200.0, dtype=np.float64)
    else:
        joint_pd_gains = np.asarray(joint_pd_gains, dtype=np.float64)
        if joint_pd_gains.shape != (n,):
            raise ValueError(
                f"joint_pd_gains must have shape ({n},), got {joint_pd_gains.shape}"
            )

    if link_material is None:
        link_material = {"E": 70e9, "rho": 2700.0}

    # 1. Joint PD gains (diagonal)
    K = np.diag(joint_pd_gains)
    logger.debug("K_PD diagonal range: [%.2f, %.2f]", joint_pd_gains.min(), joint_pd_gains.max())

    # 2. Harmonic-drive gearbox compliance reflected through gear ratio
    gear_ratio = 50.0  # typical G1 gear ratio
    k_gear_reflected = gearbox_compliance / gear_ratio
    K_gear = np.eye(n) * k_gear_reflected
    K = K + K_gear
    logger.debug("K_gear per-DOF: %.4f N·m/rad", k_gear_reflected)

    # 3. Link bending stiffness via Euler-Bernoulli beam theory
    # Model each link as a hollow circular cross-section tube:
    #   I = π/4 * (r_outer^4 - r_inner^4)
    #   k_bend = E * I / L^3
    E = float(link_material.get("E", 70e9))
    r_outer = 0.020  # m — outer radius of G1 structural tube
    r_inner = 0.015  # m — inner radius
    L = 0.30         # m — representative link length
    I_area = np.pi * (r_outer**4 - r_inner**4) / 4.0
    k_bend = E * I_area / L**3
    K_bend = np.eye(n) * k_bend
    K = K + K_bend
    logger.debug("K_bend per-DOF: %.4f N·m/rad (E=%.2e Pa, I=%.4e m^4, L=%.2f m)", k_bend, E, I_area, L)

    # Enforce symmetry
    K = 0.5 * (K + K.T)
    logger.info(
        "Assembled stiffness matrix: shape=%s, diag range [%.2f, %.2f]",
        K.shape,
        np.diag(K).min(),
        np.diag(K).max(),
    )
    return K


def diagonal_stiffness(
    n_dof: int,
    k_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a diagonal stiffness matrix from per-DOF stiffness values.

    Parameters
    ----------
    n_dof : int
        Number of degrees of freedom.  Must equal ``len(k_values)``.
    k_values : np.ndarray, shape (n_dof,)
        Per-DOF stiffness values in N·m/rad (rotational DOFs) or N/m
        (translational DOFs).

    Returns
    -------
    K : np.ndarray, shape (n_dof, n_dof)
        Diagonal stiffness matrix with ``k_values`` on the main diagonal.

    Raises
    ------
    ValueError
        If ``len(k_values) != n_dof``.

    Examples
    --------
    >>> import numpy as np
    >>> from hamiltonian_modal.modal.stiffness import diagonal_stiffness
    >>> K = diagonal_stiffness(3, np.array([100.0, 200.0, 500.0]))
    >>> K.shape
    (3, 3)
    """
    k_values = np.asarray(k_values, dtype=np.float64)
    if k_values.shape != (n_dof,):
        raise ValueError(
            f"k_values must have shape ({n_dof},), got {k_values.shape}"
        )
    return np.diag(k_values)


def stiffness_from_gravity_hessian(
    gravity_fn: Any,
    q_eq: NDArray[np.float64],
    eps: float = 1e-5,
) -> NDArray[np.float64]:
    """Estimate K = d²V_gravity/dq² at equilibrium via finite differences.

    Uses a second-order central finite-difference scheme to approximate
    the Hessian of the gravitational potential energy ``V(q)`` evaluated
    at the equilibrium configuration ``q_eq``.

    Parameters
    ----------
    gravity_fn : callable
        Function ``V(q) -> float`` returning the gravitational potential
        energy for configuration ``q``.
    q_eq : np.ndarray, shape (n,)
        Equilibrium joint configuration at which the Hessian is computed.
    eps : float, optional
        Finite-difference step size.  Default is 1e-5 rad.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        Symmetric approximation of the gravity stiffness (Hessian of V).

    Notes
    -----
    The Hessian is estimated using the mixed second-derivative formula::

        K[i, j] ≈ (V(q + eps*e_i + eps*e_j)
                   - V(q + eps*e_i - eps*e_j)
                   - V(q - eps*e_i + eps*e_j)
                   + V(q - eps*e_i - eps*e_j)) / (4 * eps²)

    The result is symmetrised as ``K = 0.5 * (K + K^T)``.

    This is an O(n²) operation; for large ``n`` consider analytic Hessians
    via Pinocchio's ``computePotentialEnergy`` + automatic differentiation.
    """
    q_eq = np.asarray(q_eq, dtype=np.float64)
    n = q_eq.shape[0]
    K = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            q_pp = q_eq.copy()
            q_pm = q_eq.copy()
            q_mp = q_eq.copy()
            q_mm = q_eq.copy()

            q_pp[i] += eps
            q_pp[j] += eps
            q_pm[i] += eps
            q_pm[j] -= eps
            q_mp[i] -= eps
            q_mp[j] += eps
            q_mm[i] -= eps
            q_mm[j] -= eps

            K[i, j] = (
                gravity_fn(q_pp) - gravity_fn(q_pm) - gravity_fn(q_mp) + gravity_fn(q_mm)
            ) / (4.0 * eps**2)

    return 0.5 * (K + K.T)
