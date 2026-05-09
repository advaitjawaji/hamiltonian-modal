"""Jumping environment interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig
from hamiltonian_modal.envs.g1_base import EnvState, G1BaseEnv

__all__ = ["G1JumpingEnv"]

_N_DOF = 23
_OBS_DIM = _N_DOF * 2
_ACT_DIM = _N_DOF


class G1JumpingEnv(G1BaseEnv):
    """Jumping environment for the Unitree G1.

    Reward: peak height attained (proxy: maximum upward velocity).
    Termination: joint limits exceeded.
    """

    _Q_LIMIT: float = 3.14

    def __init__(self, config: EnvConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self._max_upward_v: float = 0.0

    def _reset_state(self) -> EnvState:
        self._max_upward_v = 0.0
        return super()._reset_state()

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
        return np.concatenate([state.q, state.v])

    def _compute_reward(
        self,
        state: EnvState,
        action: NDArray[np.float64],
        next_state: EnvState,
    ) -> float:
        upward_v = float(next_state.v[-1])  # last DoF as height proxy
        if upward_v > self._max_upward_v:
            self._max_upward_v = upward_v
        torque_penalty = 0.001 * float(np.sum(action**2))
        return upward_v - torque_penalty

    def _is_terminated(self, state: EnvState) -> bool:
        return bool(np.any(np.abs(state.q) > self._Q_LIMIT))
