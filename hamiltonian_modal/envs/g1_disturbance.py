"""Disturbance benchmark environment interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig
from hamiltonian_modal.envs.g1_base import EnvState, G1BaseEnv

__all__ = ["G1DisturbanceEnv"]

_N_DOF = 23
_OBS_DIM = _N_DOF * 2 + 3
_ACT_DIM = _N_DOF


class G1DisturbanceEnv(G1BaseEnv):
    """Continuous random-disturbance benchmark environment.

    At every step a random force is sampled from a Gaussian and applied to
    the first three DoFs.  The agent must maintain upright posture.

    This environment is intended for benchmarking robustness rather than
    training; use :class:`~hamiltonian_modal.envs.g1_push_recovery.G1PushRecoveryEnv`
    for episodic push-recovery training.
    """

    _Q_LIMIT: float = 1.5

    def __init__(
        self,
        config: EnvConfig | None = None,
        seed: int = 0,
        disturbance_std: float = 5.0,
    ) -> None:
        super().__init__(config, seed)
        self._disturbance_std = disturbance_std
        self._disturbance: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

    @property
    def n_dof(self) -> int:
        return _N_DOF

    @property
    def obs_dim(self) -> int:
        return _OBS_DIM

    @property
    def action_dim(self) -> int:
        return _ACT_DIM

    def _compute_obs(self, state: EnvState) -> NDArray[np.float64]:
        return np.concatenate([state.q, state.v, self._disturbance])

    def _compute_reward(
        self,
        state: EnvState,
        action: NDArray[np.float64],
        next_state: EnvState,
    ) -> float:
        balance = -float(np.sum(next_state.q**2))
        torque_penalty = 0.001 * float(np.sum(action**2))
        return balance - torque_penalty

    def _is_terminated(self, state: EnvState) -> bool:
        return bool(np.any(np.abs(state.q) > self._Q_LIMIT))

    def _dynamics(self, state: EnvState, action: NDArray[np.float64]) -> EnvState:
        self._disturbance = self._rng.standard_normal(3) * self._disturbance_std
        dt = self.config.dt
        perturbed = action.copy()
        perturbed[:3] += self._disturbance * dt
        new_v = state.v + perturbed * dt
        new_q = state.q + new_v * dt
        return EnvState(q=new_q, v=new_v, t=state.t + dt, step=state.step + 1)
