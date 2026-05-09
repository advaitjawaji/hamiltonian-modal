"""Base environment interfaces for Unitree G1.

Defines the abstract :class:`G1BaseEnv` that all concrete G1 environments
inherit from.  The interface follows a gym-like ``reset() / step()`` contract
using plain NumPy arrays so no RL framework dependency is required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import EnvConfig

__all__ = [
    "EnvState",
    "StepResult",
    "G1BaseEnv",
]


@dataclass
class EnvState:
    """Snapshot of the G1 environment state.

    Attributes
    ----------
    q:
        Joint positions, shape ``(n_dof,)``.
    v:
        Joint velocities, shape ``(n_dof,)``.
    t:
        Elapsed simulation time in seconds.
    step:
        Current step count within the episode.
    """

    q: NDArray[np.float64]
    v: NDArray[np.float64]
    t: float = 0.0
    step: int = 0


@dataclass
class StepResult:
    """Result returned by :meth:`G1BaseEnv.step`.

    Attributes
    ----------
    obs:
        Observation vector, shape ``(obs_dim,)``.
    reward:
        Scalar reward.
    terminated:
        Whether the episode ended due to a terminal condition (fall, etc.).
    truncated:
        Whether the episode ended due to step-count limit.
    info:
        Auxiliary diagnostic information.
    """

    obs: NDArray[np.float64]
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class G1BaseEnv(ABC):
    """Abstract base class for Unitree G1 environments.

    Subclasses must implement :meth:`_compute_obs`, :meth:`_compute_reward`,
    and :meth:`_is_terminated`.  Optionally override :meth:`_reset_state` to
    customise the initial joint configuration.

    Parameters
    ----------
    config:
        Environment configuration.
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int = 0) -> None:
        self.config = config or EnvConfig()
        self._rng = np.random.default_rng(seed)
        self._state: EnvState | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of the observation vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action vector."""

    @abstractmethod
    def _compute_obs(self, state: EnvState) -> NDArray[np.float64]:
        """Compute the observation from *state*."""

    @abstractmethod
    def _compute_reward(
        self, state: EnvState, action: NDArray[np.float64], next_state: EnvState
    ) -> float:
        """Compute the scalar reward for a transition."""

    @abstractmethod
    def _is_terminated(self, state: EnvState) -> bool:
        """Return ``True`` when the episode should end due to failure."""

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def _reset_state(self) -> EnvState:
        """Return the initial :class:`EnvState` for a new episode.

        The default implementation sets q and v to zero.  Override to sample
        from a distribution of initial conditions.
        """
        return EnvState(
            q=np.zeros(self.n_dof, dtype=np.float64),
            v=np.zeros(self.n_dof, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Public gym-like API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float64], dict]:
        """Reset the environment and return (observation, info).

        Parameters
        ----------
        seed:
            Optional new seed.  Passed to the internal RNG.

        Returns
        -------
        tuple[NDArray, dict]
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._reset_state()
        obs = self._compute_obs(self._state)
        if self.config.obs_noise_std > 0.0:
            obs = obs + self._rng.standard_normal(obs.shape) * self.config.obs_noise_std
        return obs, {}

    def step(self, action: NDArray[np.float64]) -> StepResult:
        """Advance the environment by one step.

        Parameters
        ----------
        action:
            Action vector, shape ``(action_dim,)``.

        Returns
        -------
        StepResult
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        scaled = np.clip(action, -1.0, 1.0) * self.config.action_scale
        next_state = self._dynamics(self._state, scaled)
        reward = self._compute_reward(self._state, scaled, next_state)
        self._state = next_state
        obs = self._compute_obs(self._state)
        if self.config.obs_noise_std > 0.0:
            obs = obs + self._rng.standard_normal(obs.shape) * self.config.obs_noise_std
        terminated = self._is_terminated(self._state)
        truncated = self._state.step >= self.config.episode_length
        return StepResult(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info={})

    # ------------------------------------------------------------------
    # Internal dynamics
    # ------------------------------------------------------------------

    def _dynamics(self, state: EnvState, action: NDArray[np.float64]) -> EnvState:
        """Euler integration: advance state by one timestep using *action* as acceleration.

        Subclasses may override to plug in a differentiable simulator or a
        Hamiltonian integrator.
        """
        dt = self.config.dt
        new_v = state.v + action * dt
        new_q = state.q + new_v * dt
        return EnvState(q=new_q, v=new_v, t=state.t + dt, step=state.step + 1)
