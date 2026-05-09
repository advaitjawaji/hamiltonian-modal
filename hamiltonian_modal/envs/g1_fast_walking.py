"""Fast-walking environment interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig
from hamiltonian_modal.envs.g1_base import EnvState, G1BaseEnv

__all__ = ["G1FastWalkingEnv"]

_N_DOF = 23
_OBS_DIM = _N_DOF * 2
_ACT_DIM = _N_DOF


class G1FastWalkingEnv(G1BaseEnv):
    """High-speed walking environment for the Unitree G1.

    Similar to :class:`~hamiltonian_modal.envs.g1_walking.G1WalkingEnv` but
    rewards higher cadence and penalises slower forward progress more strongly.
    """

    _Q_LIMIT: float = 3.14
    _TARGET_SPEED: float = 2.0  # m s⁻¹ (proxy)

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
        forward_vel = float(next_state.v[0])
        speed_reward = -abs(forward_vel - self._TARGET_SPEED)
        torque_penalty = 0.001 * float(np.sum(action**2))
        return speed_reward - torque_penalty

    def _is_terminated(self, state: EnvState) -> bool:
        return bool(np.any(np.abs(state.q) > self._Q_LIMIT))
