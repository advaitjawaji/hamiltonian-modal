"""Differential Dynamic Programming (DDP) in modal coordinates.

DDP extends iLQR by including the second-order (Hessian) terms in the
dynamics linearisation, leading to faster local convergence near
optimal trajectories.

Reference: Jacobson & Mayne, "Differential Dynamic Programming" (1970).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.mpc.ilqr import ILQRConfig, ILQRResult, ILQRSolver

logger = logging.getLogger(__name__)


@dataclass
class DDPConfig(ILQRConfig):
    """Configuration for the DDP solver.

    Inherits all iLQR settings and adds DDP-specific parameters.

    Parameters
    ----------
    use_second_order_dynamics : bool — if True, include Hessian correction
        terms (full DDP); if False, reduces to iLQR
    reg_min : float — minimum regularisation
    reg_max : float — maximum regularisation before aborting
    reg_factor : float — factor to increase/decrease regularisation
    """
    use_second_order_dynamics: bool = True
    reg_min: float = 1e-6
    reg_max: float = 1e3
    reg_factor: float = 10.0


class DDPSolver(ILQRSolver):
    """Differential Dynamic Programming solver.

    Extends ILQRSolver with optional second-order dynamics corrections
    and adaptive regularisation for improved convergence.

    Parameters
    ----------
    dynamics : callable — f(x, u, m) → x_next
    cost_fn : callable — l(x, u) → float
    terminal_cost_fn : callable — l_f(x) → float
    config : DDPConfig
    state_dim : int
    action_dim : int
    contact_mode : int
    eps : float — finite-difference step
    """

    def __init__(
        self,
        dynamics: Callable[[NDArray, NDArray, int], NDArray],
        cost_fn: Callable[[NDArray, NDArray], float],
        terminal_cost_fn: Callable[[NDArray], float],
        config: "DDPConfig | None" = None,
        state_dim: int = 20,
        action_dim: int = 23,
        contact_mode: int = 0,
        eps: float = 1e-5,
    ) -> None:
        ddp_cfg = config or DDPConfig()
        super().__init__(
            dynamics=dynamics,
            cost_fn=cost_fn,
            terminal_cost_fn=terminal_cost_fn,
            config=ddp_cfg,
            state_dim=state_dim,
            action_dim=action_dim,
            contact_mode=contact_mode,
            eps=eps,
        )
        self._ddp_config = ddp_cfg
        self._reg = ddp_cfg.reg

    def _compute_dynamics_hessian(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute Hessian corrections Fxx, Fuu, Fux contracted with value grad v.

        For efficiency, uses a diagonal finite-difference approximation.
        Full cross-terms are zeroed (acceptable for near-quadratic dynamics).

        Parameters
        ----------
        x : shape (n,) — state
        u : shape (m,) — action
        v : shape (n,) — value function gradient

        Returns
        -------
        Fxx_v : shape (n, n)
        Fuu_v : shape (m, m)
        Fux_v : shape (m, n)
        """
        n, m = self.state_dim, self.action_dim
        eps = self.eps
        f0 = self.dynamics(x, u, self.contact_mode)

        Fxx_v = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            d2f = (
                self.dynamics(xp, u, self.contact_mode)
                - 2.0 * f0
                + self.dynamics(xm, u, self.contact_mode)
            ) / (eps ** 2)
            Fxx_v[i, i] = float(v @ d2f)

        Fuu_v = np.zeros((m, m), dtype=np.float64)
        for j in range(m):
            up = u.copy(); up[j] += eps
            um = u.copy(); um[j] -= eps
            d2f = (
                self.dynamics(x, up, self.contact_mode)
                - 2.0 * f0
                + self.dynamics(x, um, self.contact_mode)
            ) / (eps ** 2)
            Fuu_v[j, j] = float(v @ d2f)

        Fux_v = np.zeros((m, n), dtype=np.float64)  # cross-term zeroed

        return Fxx_v, Fuu_v, Fux_v

    def _backward_pass_ddp(
        self,
        xs: NDArray[np.float64],
        us: NDArray[np.float64],
    ) -> tuple[list[NDArray], list[NDArray], bool]:
        """DDP backward pass with second-order dynamics corrections.

        Returns (Ks, ks, success).  If Q_uu is not positive definite,
        increases regularisation and returns success=False.
        """
        T = self.config.horizon
        use_2nd = self._ddp_config.use_second_order_dynamics

        Vx, Vxx = self._quadraticise_terminal(xs[T])
        Ks: list[NDArray] = [np.zeros((self.action_dim, self.state_dim))] * T
        ks: list[NDArray] = [np.zeros(self.action_dim)] * T

        for t in reversed(range(T)):
            A, B = self._linearise_dynamics(xs[t], us[t])
            lx, lu, lxx, luu, lux = self._quadraticise_cost(xs[t], us[t])

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B + self._reg * np.eye(self.action_dim)
            Qux = lux + B.T @ Vxx @ A

            if use_2nd:
                Fxx_v, Fuu_v, Fux_v = self._compute_dynamics_hessian(xs[t], us[t], Vx)
                Qxx = Qxx + Fxx_v
                Quu = Quu + Fuu_v
                Qux = Qux + Fux_v

            # Check positive definiteness of Q_uu
            eigvals = np.linalg.eigvalsh(Quu)
            if float(np.min(eigvals)) <= 0.0:
                return Ks, ks, False  # Not PD — caller should increase reg

            try:
                Quu_inv = np.linalg.inv(Quu)
            except np.linalg.LinAlgError:
                return Ks, ks, False

            K = -Quu_inv @ Qux
            k = -Quu_inv @ Qu
            Ks[t] = K
            ks[t] = k

            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

        return Ks, ks, True

    def solve(
        self,
        x0: NDArray[np.float64],
        u_init: "NDArray[np.float64] | None" = None,
    ) -> ILQRResult:
        """Solve the trajectory optimisation problem using DDP.

        Parameters
        ----------
        x0 : shape (state_dim,)
        u_init : shape (T, action_dim) or None

        Returns
        -------
        result : ILQRResult
        """
        T = self.config.horizon
        x0 = np.asarray(x0, dtype=np.float64)

        if u_init is None:
            us = np.zeros((T, self.action_dim), dtype=np.float64)
        else:
            us = np.asarray(u_init, dtype=np.float64).copy()

        xs, cost = self._forward_pass(x0, us)
        converged = False
        self._reg = self._ddp_config.reg

        for it in range(self.config.n_iterations):
            # Backward pass with adaptive regularisation
            success = False
            for _ in range(10):
                Ks, ks, success = self._backward_pass_ddp(xs, us)
                if success:
                    break
                self._reg = min(self._reg * self._ddp_config.reg_factor, self._ddp_config.reg_max)
                if self._reg >= self._ddp_config.reg_max:
                    logger.warning("DDP: max regularisation reached")
                    break

            if not success:
                break

            # Forward pass with line search
            alpha = self.config.alpha
            improved = False
            while alpha >= self.config.min_alpha:
                us_new = np.zeros_like(us)
                xs_new = np.zeros_like(xs)
                xs_new[0] = x0.copy()
                for t in range(T):
                    delta_x = xs_new[t] - xs[t]
                    us_new[t] = us[t] + Ks[t] @ delta_x + alpha * ks[t]
                    xs_new[t + 1] = self.dynamics(xs_new[t], us_new[t], self.contact_mode)

                cost_new = sum(
                    self.cost_fn(xs_new[t], us_new[t]) for t in range(T)
                ) + self.terminal_cost_fn(xs_new[T])

                if cost_new < cost:
                    improved = True
                    break
                alpha *= self.config.alpha_decay

            if not improved:
                logger.debug("DDP: line search failed at iteration %d", it)
                break

            improvement = cost - cost_new
            xs, us, cost = xs_new, us_new, cost_new
            # Decrease regularisation on success
            self._reg = max(self._reg / self._ddp_config.reg_factor, self._ddp_config.reg_min)

            if improvement < self.config.convergence_tol:
                converged = True
                break

        return ILQRResult(
            us=us,
            xs=xs,
            cost=float(cost),
            n_iterations=it + 1,
            converged=converged,
        )
