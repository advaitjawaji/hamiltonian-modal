"""Iterative Linear Quadratic Regulator (iLQR) in modal coordinates.

Solves the trajectory optimisation problem:
  min_{u_0,...,u_{T-1}} Σ_t l(x_t, u_t) + l_f(x_T)
subject to x_{t+1} = f(x_t, u_t)

where x = [η; η̇] is the modal state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ILQRConfig:
    """Configuration for the iLQR solver.

    Parameters
    ----------
    horizon : int — planning horizon T
    n_iterations : int — number of forward-backward pass iterations
    alpha : float — initial line-search step size
    alpha_decay : float — line-search backtracking factor
    min_alpha : float — minimum line-search step
    reg : float — regularisation added to Q_uu for numerical stability
    convergence_tol : float — cost improvement threshold for early stopping
    """
    horizon: int = 20
    n_iterations: int = 10
    alpha: float = 1.0
    alpha_decay: float = 0.5
    min_alpha: float = 1e-4
    reg: float = 1e-4
    convergence_tol: float = 1e-6


@dataclass
class ILQRResult:
    """Result of an iLQR solve.

    Attributes
    ----------
    us : shape (T, action_dim) — optimal control sequence
    xs : shape (T+1, state_dim) — state trajectory
    cost : float — final total cost
    n_iterations : int — iterations used
    converged : bool
    """
    us: NDArray[np.float64]
    xs: NDArray[np.float64]
    cost: float
    n_iterations: int
    converged: bool


class ILQRSolver:
    """Iterative LQR trajectory optimiser.

    Uses finite differences to linearise the dynamics and quadraticise
    the cost at each timestep, then performs a backward Riccati sweep
    to compute feedback gains and a forward pass with line search.

    Parameters
    ----------
    dynamics : callable — f(x, u, m) → x_next
    cost_fn : callable — l(x, u) → float
    terminal_cost_fn : callable — l_f(x) → float
    config : ILQRConfig
    state_dim : int
    action_dim : int
    contact_mode : int
    eps : float — finite-difference perturbation for linearisation
    """

    def __init__(
        self,
        dynamics: Callable[[NDArray, NDArray, int], NDArray],
        cost_fn: Callable[[NDArray, NDArray], float],
        terminal_cost_fn: Callable[[NDArray], float],
        config: "ILQRConfig | None" = None,
        state_dim: int = 20,
        action_dim: int = 23,
        contact_mode: int = 0,
        eps: float = 1e-5,
    ) -> None:
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.config = config or ILQRConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.contact_mode = contact_mode
        self.eps = eps

    # ------------------------------------------------------------------
    # Finite-difference linearisation
    # ------------------------------------------------------------------

    def _linearise_dynamics(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute (A, B) = (∂f/∂x, ∂f/∂u) via central FD."""
        n, m = self.state_dim, self.action_dim
        x_nom = self.dynamics(x, u, self.contact_mode)

        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            xp = x.copy(); xp[i] += self.eps
            xm = x.copy(); xm[i] -= self.eps
            A[:, i] = (self.dynamics(xp, u, self.contact_mode) - self.dynamics(xm, u, self.contact_mode)) / (2 * self.eps)

        B = np.zeros((n, m), dtype=np.float64)
        for j in range(m):
            up = u.copy(); up[j] += self.eps
            um = u.copy(); um[j] -= self.eps
            B[:, j] = (self.dynamics(x, up, self.contact_mode) - self.dynamics(x, um, self.contact_mode)) / (2 * self.eps)

        return A, B

    def _quadraticise_cost(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Compute (lx, lu, lxx, luu, lux) via central FD."""
        n, m = self.state_dim, self.action_dim
        eps = self.eps

        l0 = self.cost_fn(x, u)

        # First-order terms
        lx = np.zeros(n, dtype=np.float64)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            lx[i] = (self.cost_fn(xp, u) - self.cost_fn(xm, u)) / (2 * eps)

        lu = np.zeros(m, dtype=np.float64)
        for j in range(m):
            up = u.copy(); up[j] += eps
            um = u.copy(); um[j] -= eps
            lu[j] = (self.cost_fn(x, up) - self.cost_fn(x, um)) / (2 * eps)

        # Second-order terms (diagonal FD approximation for speed)
        lxx = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            lxx[i, i] = (self.cost_fn(xp, u) - 2 * l0 + self.cost_fn(xm, u)) / (eps ** 2)

        luu = np.zeros((m, m), dtype=np.float64)
        for j in range(m):
            up = u.copy(); up[j] += eps
            um = u.copy(); um[j] -= eps
            luu[j, j] = (self.cost_fn(x, up) - 2 * l0 + self.cost_fn(x, um)) / (eps ** 2)

        lux = np.zeros((m, n), dtype=np.float64)  # cross term (zeroed for simplicity)

        return lx, lu, lxx, luu, lux

    def _quadraticise_terminal(
        self,
        x: NDArray[np.float64],
    ) -> tuple[NDArray, NDArray]:
        """Compute (lx_f, lxx_f) for terminal cost."""
        n = self.state_dim
        eps = self.eps
        l0 = self.terminal_cost_fn(x)

        lx_f = np.zeros(n, dtype=np.float64)
        lxx_f = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            lx_f[i] = (self.terminal_cost_fn(xp) - self.terminal_cost_fn(xm)) / (2 * eps)
            lxx_f[i, i] = (self.terminal_cost_fn(xp) - 2 * l0 + self.terminal_cost_fn(xm)) / (eps ** 2)

        return lx_f, lxx_f

    # ------------------------------------------------------------------
    # Forward / backward passes
    # ------------------------------------------------------------------

    def _forward_pass(
        self,
        x0: NDArray[np.float64],
        us: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        """Roll out dynamics given control sequence, return (xs, cost)."""
        T = self.config.horizon
        xs = np.zeros((T + 1, self.state_dim), dtype=np.float64)
        xs[0] = x0.copy()
        total_cost = 0.0
        for t in range(T):
            xs[t + 1] = self.dynamics(xs[t], us[t], self.contact_mode)
            total_cost += self.cost_fn(xs[t], us[t])
        total_cost += self.terminal_cost_fn(xs[T])
        return xs, total_cost

    def _backward_pass(
        self,
        xs: NDArray[np.float64],
        us: NDArray[np.float64],
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Backward Riccati sweep. Returns (Ks, ks) feedback gains."""
        T = self.config.horizon
        reg = self.config.reg

        Vx, Vxx = self._quadraticise_terminal(xs[T])

        Ks: list[NDArray[np.float64]] = [np.zeros((self.action_dim, self.state_dim))] * T
        ks: list[NDArray[np.float64]] = [np.zeros(self.action_dim)] * T

        for t in reversed(range(T)):
            A, B = self._linearise_dynamics(xs[t], us[t])
            lx, lu, lxx, luu, lux = self._quadraticise_cost(xs[t], us[t])

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B + reg * np.eye(self.action_dim)
            Qux = lux + B.T @ Vxx @ A

            try:
                Quu_inv = np.linalg.inv(Quu)
            except np.linalg.LinAlgError:
                Quu_inv = np.linalg.pinv(Quu)

            K = -Quu_inv @ Qux
            k = -Quu_inv @ Qu

            Ks[t] = K
            ks[t] = k

            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K

        return Ks, ks

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(
        self,
        x0: NDArray[np.float64],
        u_init: "NDArray[np.float64] | None" = None,
    ) -> ILQRResult:
        """Solve the trajectory optimisation problem from x0.

        Parameters
        ----------
        x0 : shape (state_dim,) — initial state
        u_init : shape (T, action_dim) or None — warm-start control sequence

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
        it = 0

        for it in range(self.config.n_iterations):
            Ks, ks = self._backward_pass(xs, us)

            # Line search
            alpha = self.config.alpha
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
                    break
                alpha *= self.config.alpha_decay
            else:
                logger.debug("iLQR: line search failed at iteration %d", it)
                break

            improvement = cost - cost_new
            xs, us, cost = xs_new, us_new, cost_new

            if improvement < self.config.convergence_tol:
                converged = True
                break

        return ILQRResult(
            us=us,
            xs=xs,
            cost=cost,
            n_iterations=it + 1,
            converged=converged,
        )
