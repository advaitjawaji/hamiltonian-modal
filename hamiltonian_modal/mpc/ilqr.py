"""Iterative LQR interfaces.

Implements the *iterative Linear Quadratic Regulator* (iLQR) for trajectory
optimisation.  Given a differentiable dynamics model and a quadratic-ish cost,
iLQR iteratively linearises the dynamics and quadraticises the cost around the
current nominal trajectory to compute a locally optimal policy update.

The implementation follows the standard backward-pass (Riccati) + forward-pass
(line-search) algorithm using only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import ILQRConfig

__all__ = [
    "ILQRResult",
    "ILQR",
]

DynamicsFn = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
CostFn = Callable[
    [NDArray[np.float64], NDArray[np.float64], int],
    float,
]
TerminalCostFn = Callable[[NDArray[np.float64]], float]


@dataclass
class ILQRResult:
    """Result of an iLQR optimisation.

    Attributes
    ----------
    xs:
        Optimised state trajectory, shape ``(T+1, n_state)``.
    us:
        Optimised control sequence, shape ``(T, n_ctrl)``.
    total_cost:
        Total cost of the optimised trajectory.
    n_iterations:
        Number of iLQR iterations performed.
    """

    xs: NDArray[np.float64]
    us: NDArray[np.float64]
    total_cost: float
    n_iterations: int


class ILQR:
    """Iterative Linear Quadratic Regulator.

    Parameters
    ----------
    config:
        :class:`~hamiltonian_modal.config.ILQRConfig`.
    dynamics_fn:
        Callable ``(x, u) → x_next``.
    cost_fn:
        Callable ``(x, u, t) → scalar_cost`` for running cost.
    terminal_cost_fn:
        Callable ``x_T → scalar_cost`` for terminal cost.
    n_state:
        Dimensionality of the state vector.
    n_ctrl:
        Dimensionality of the control vector.
    fd_eps:
        Finite-difference step for Jacobian estimation.
    """

    def __init__(
        self,
        config: ILQRConfig,
        dynamics_fn: DynamicsFn,
        cost_fn: CostFn,
        terminal_cost_fn: TerminalCostFn,
        n_state: int,
        n_ctrl: int,
        fd_eps: float = 1e-4,
    ) -> None:
        self.config = config
        self._f = dynamics_fn
        self._l = cost_fn
        self._lf = terminal_cost_fn
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.fd_eps = fd_eps

    # ------------------------------------------------------------------
    # Finite-difference Jacobians
    # ------------------------------------------------------------------

    def _jacobians(
        self, x: NDArray[np.float64], u: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute ∂f/∂x and ∂f/∂u via central differences."""
        eps = self.fd_eps
        Fx = np.zeros((self.n_state, self.n_state))
        Fu = np.zeros((self.n_state, self.n_ctrl))
        for i in range(self.n_state):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            Fx[:, i] = (self._f(xp, u) - self._f(xm, u)) / (2 * eps)
        for i in range(self.n_ctrl):
            up, um = u.copy(), u.copy()
            up[i] += eps
            um[i] -= eps
            Fu[:, i] = (self._f(x, up) - self._f(x, um)) / (2 * eps)
        return Fx, Fu

    def _cost_gradients(
        self, x: NDArray[np.float64], u: NDArray[np.float64], t: int
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Compute first- and second-order cost derivatives via FD."""
        eps = self.fd_eps
        l0 = self._l(x, u, t)
        lx = np.zeros(self.n_state)
        lu = np.zeros(self.n_ctrl)
        for i in range(self.n_state):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            lx[i] = (self._l(xp, u, t) - self._l(xm, u, t)) / (2 * eps)
        for i in range(self.n_ctrl):
            up, um = u.copy(), u.copy()
            up[i] += eps
            um[i] -= eps
            lu[i] = (self._l(x, up, t) - self._l(x, um, t)) / (2 * eps)
        lxx = np.eye(self.n_state) * 1e-4   # regularised identity
        luu = np.eye(self.n_ctrl) * 1e-4
        lux = np.zeros((self.n_ctrl, self.n_state))
        return lx, lu, lxx, luu, lux

    # ------------------------------------------------------------------
    # iLQR passes
    # ------------------------------------------------------------------

    def _rollout(
        self, x0: NDArray[np.float64], us: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        T = us.shape[0]
        xs = np.zeros((T + 1, self.n_state))
        xs[0] = x0
        total_cost = 0.0
        for t in range(T):
            xs[t + 1] = self._f(xs[t], us[t])
            total_cost += self._l(xs[t], us[t], t)
        total_cost += self._lf(xs[T])
        return xs, total_cost

    def _backward(
        self,
        xs: NDArray[np.float64],
        us: NDArray[np.float64],
        reg: float,
    ) -> tuple[list, list]:
        T = us.shape[0]
        # Terminal value
        xT = xs[T]
        eps = self.fd_eps
        Vx = np.zeros(self.n_state)
        for i in range(self.n_state):
            xp, xm = xT.copy(), xT.copy()
            xp[i] += eps
            xm[i] -= eps
            Vx[i] = (self._lf(xp) - self._lf(xm)) / (2 * eps)
        Vxx = np.eye(self.n_state) * 1e-4

        ks, Ks = [], []
        for t in reversed(range(T)):
            Fx, Fu = self._jacobians(xs[t], us[t])
            lx, lu, lxx, luu, lux = self._cost_gradients(xs[t], us[t], t)
            Qx = lx + Fx.T @ Vx
            Qu = lu + Fu.T @ Vx
            Qxx = lxx + Fx.T @ Vxx @ Fx
            Quu = luu + Fu.T @ Vxx @ Fu + reg * np.eye(self.n_ctrl)
            Qux = lux + Fu.T @ Vxx @ Fx
            Quu_inv = np.linalg.inv(Quu)
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux
            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)
            ks.insert(0, k)
            Ks.insert(0, K)
        return ks, Ks

    def optimise(
        self,
        x0: NDArray[np.float64],
        us_init: NDArray[np.float64] | None = None,
    ) -> ILQRResult:
        """Run iLQR from *x0*.

        Parameters
        ----------
        x0:
            Initial state, shape ``(n_state,)``.
        us_init:
            Initial control sequence, shape ``(horizon, n_ctrl)``.
            Defaults to zeros.

        Returns
        -------
        ILQRResult
        """
        T = self.config.horizon
        us = (
            np.zeros((T, self.n_ctrl), dtype=np.float64)
            if us_init is None
            else us_init.copy()
        )
        xs, cost = self._rollout(x0, us)
        reg = self.config.reg_init

        for iteration in range(self.config.n_iterations):
            ks, Ks = self._backward(xs, us, reg)
            # Forward pass with line search
            alpha = 1.0
            for _ in range(10):
                us_new = us.copy()
                xs_new = np.zeros_like(xs)
                xs_new[0] = x0
                for t in range(T):
                    dx = xs_new[t] - xs[t]
                    us_new[t] = us[t] + alpha * ks[t] + Ks[t] @ dx
                    xs_new[t + 1] = self._f(xs_new[t], us_new[t])
                cost_new = sum(
                    self._l(xs_new[t], us_new[t], t) for t in range(T)
                ) + self._lf(xs_new[T])
                if cost_new < cost:
                    xs, us, cost = xs_new, us_new, cost_new
                    reg = max(self.config.reg_min, reg * self.config.line_search_beta)
                    break
                alpha *= self.config.line_search_beta
            else:
                reg = min(self.config.reg_max, reg / self.config.line_search_beta)

        return ILQRResult(xs=xs, us=us, total_cost=cost, n_iterations=iteration + 1)
