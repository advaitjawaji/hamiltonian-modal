"""Flat-ground walking environment interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig
from hamiltonian_modal.envs.g1_base import EnvState, G1BaseEnv

__all__ = ["G1WalkingEnv"]

_N_DOF = 23  # Unitree G1 actuated DoFs
_OBS_DIM = _N_DOF * 2  # q + v
_ACT_DIM = _N_DOF


class G1WalkingEnv(G1BaseEnv):
    """Flat-ground walking environment for the Unitree G1.

    Reward: positive for forward velocity, penalty for large joint torques.
    Termination: joint position out of limits.
    """

    _Q_LIMIT: float = 3.14  # ~π rad

    def __init__(self, config: EnvConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)

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
        # Forward velocity proxy: change in first DoF (hip pitch)
        forward_vel = float(next_state.v[0])
        torque_penalty = 0.001 * float(np.sum(action**2))
        return forward_vel - torque_penalty

    def _is_terminated(self, state: EnvState) -> bool:
        return bool(np.any(np.abs(state.q) > self._Q_LIMIT))
