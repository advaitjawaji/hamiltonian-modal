"""Differential dynamic programming interfaces.

Extends :class:`~hamiltonian_modal.mpc.ilqr.ILQR` with second-order
value-function terms (the full DDP expansion) as described in Mayne (1966).
When ``second_order_value`` is ``False`` this reduces exactly to iLQR.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import DDPConfig
from hamiltonian_modal.mpc.ilqr import ILQR, CostFn, DynamicsFn, ILQRResult, TerminalCostFn

__all__ = [
    "DDP",
]


class DDP(ILQR):
    """Differential Dynamic Programming (second-order iLQR extension).

    Parameters
    ----------
    config:
        :class:`~hamiltonian_modal.config.DDPConfig`.
    dynamics_fn, cost_fn, terminal_cost_fn, n_state, n_ctrl, fd_eps:
        Forwarded to :class:`~hamiltonian_modal.mpc.ilqr.ILQR`.
    """

    def __init__(
        self,
        config: DDPConfig,
        dynamics_fn: DynamicsFn,
        cost_fn: CostFn,
        terminal_cost_fn: TerminalCostFn,
        n_state: int,
        n_ctrl: int,
        fd_eps: float = 1e-4,
    ) -> None:
        super().__init__(
            config, dynamics_fn, cost_fn, terminal_cost_fn, n_state, n_ctrl, fd_eps
        )
        self.ddp_config: DDPConfig = config

    def _dynamics_hessian(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Finite-difference second derivatives ∂²f/∂x² and ∂²f/∂u²."""
        eps = self.fd_eps
        Fxx = np.zeros((self.n_state, self.n_state, self.n_state))
        Fuu = np.zeros((self.n_state, self.n_ctrl, self.n_ctrl))
        f0 = self._f(x, u)
        for i in range(self.n_state):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fp = self._f(xp, u)
            fm = self._f(xm, u)
            Fxx[:, i, i] = (fp - 2 * f0 + fm) / eps**2
        for i in range(self.n_ctrl):
            up, um = u.copy(), u.copy()
            up[i] += eps
            um[i] -= eps
            fp = self._f(x, up)
            fm = self._f(x, um)
            Fuu[:, i, i] = (fp - 2 * f0 + fm) / eps**2
        return Fxx, Fuu

    def _backward(
        self,
        xs: NDArray[np.float64],
        us: NDArray[np.float64],
        reg: float,
    ) -> tuple[list, list]:
        """DDP backward pass including second-order value terms when enabled."""
        if not self.ddp_config.second_order_value:
            return super()._backward(xs, us, reg)

        T = us.shape[0]
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
            Fxx, Fuu = self._dynamics_hessian(xs[t], us[t])
            # Second-order expansion: Qxx += Σ_i Vx_i * Fxx_i
            Qxx = lxx + Fx.T @ Vxx @ Fx + np.einsum("i,ijk->jk", Vx, Fxx)
            Quu = luu + Fu.T @ Vxx @ Fu + np.einsum("i,ijk->jk", Vx, Fuu) + reg * np.eye(self.n_ctrl)
            Qux = lux + Fu.T @ Vxx @ Fx
            Qx = lx + Fx.T @ Vx
            Qu = lu + Fu.T @ Vx
            Quu_inv = np.linalg.inv(Quu)
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux
            Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qux.T @ k
            Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qux.T @ K
            Vxx = 0.5 * (Vxx + Vxx.T)
            ks.insert(0, k)
            Ks.insert(0, K)
        return ks, Ks
