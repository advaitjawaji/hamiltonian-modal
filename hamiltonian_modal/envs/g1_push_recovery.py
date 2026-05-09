"""Push-recovery environment interfaces."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig
from hamiltonian_modal.envs.g1_base import EnvState, G1BaseEnv

__all__ = ["G1PushRecoveryEnv"]

_N_DOF = 23
_OBS_DIM = _N_DOF * 2 + 3  # q, v, disturbance_force
_ACT_DIM = _N_DOF


class G1PushRecoveryEnv(G1BaseEnv):
    """Push-recovery environment for the Unitree G1.

    At a random step a horizontal impulse is applied to the robot's pelvis.
    The agent must maintain balance (small q excursion) after the perturbation.
    """

    _Q_LIMIT: float = 1.5
    _PUSH_MAGNITUDE: float = 50.0  # N
    _PUSH_DURATION: int = 5  # steps

    def __init__(self, config: EnvConfig | None = None, seed: int = 0) -> None:
        super().__init__(config, seed)
        self._push_start: int = -1
        self._disturbance: NDArray[np.float64] = np.zeros(3, dtype=np.float64)

    def _reset_state(self) -> EnvState:
        state = super()._reset_state()
        # Schedule a random push between 10 and 50 % of episode
        lo = max(1, int(0.10 * self.config.episode_length))
        hi = max(lo + 1, int(0.50 * self.config.episode_length))
        self._push_start = int(self._rng.integers(lo, hi))
        self._disturbance = np.zeros(3, dtype=np.float64)
        return state

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
        # Apply disturbance as an additional acceleration on the first 3 DoFs
        if (
            self._push_start
            <= state.step
            < self._push_start + self._PUSH_DURATION
        ):
            direction = self._rng.standard_normal(3)
            direction /= np.linalg.norm(direction) + 1e-8
            self._disturbance = direction * self._PUSH_MAGNITUDE
        else:
            self._disturbance = np.zeros(3, dtype=np.float64)

        dt = self.config.dt
        perturbed_action = action.copy()
        perturbed_action[:3] += self._disturbance / max(1.0, float(np.max(np.abs(action[:3]) + 1)))
        new_v = state.v + perturbed_action * dt
        new_q = state.q + new_v * dt
        return EnvState(q=new_q, v=new_v, t=state.t + dt, step=state.step + 1)
