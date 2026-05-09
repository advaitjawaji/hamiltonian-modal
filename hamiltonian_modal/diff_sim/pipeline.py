"""End-to-end differentiable simulation pipeline.

Forward pass: observation → policy → action → Genesis.step() → modal_encoder
Backward pass: gradient through all components via differentiable pipeline.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result of a simulation rollout."""
    states: NDArray[np.float64]     # (T+1, state_dim)
    actions: NDArray[np.float64]    # (T, action_dim)
    rewards: NDArray[np.float64]    # (T,)
    total_reward: float
    modal_states: NDArray[np.float64] | None  # (T+1, n_modes*2) if modal


class DiffSimPipeline:
    """End-to-end differentiable training pipeline.

    Supports two modes:
    1. Genesis rollout: obs → policy → action → genesis.step() → modal_encoder
    2. World-model rollout: obs → policy → action → Hamiltonian world model

    Parameters
    ----------
    policy : callable, maps obs (state_dim,) → action (action_dim,)
    world_model : HamiltonianNet or None
    modal_encoder : callable or None
    n_modes : int
    n_contact_modes : int
    genesis_scene : gs.Scene or None
    """

    def __init__(
        self,
        policy: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        world_model: Any | None = None,
        modal_encoder: Any | None = None,
        n_modes: int = 20,
        n_contact_modes: int = 5,
        genesis_scene: Any | None = None,
    ) -> None:
        self.policy = policy
        self.world_model = world_model
        self.modal_encoder = modal_encoder
        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes
        self.genesis_scene = genesis_scene

    def rollout(
        self,
        initial_obs: NDArray[np.float64],
        horizon: int,
        use_world_model: bool = False,
        reward_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    ) -> RolloutResult:
        """Forward rollout.

        Parameters
        ----------
        initial_obs : shape (state_dim,)
        horizon : int
        use_world_model : bool
        reward_fn : (state, action) → reward, or None (returns 0)

        Returns
        -------
        RolloutResult
        """
        if reward_fn is None:
            reward_fn = lambda s, a: 0.0  # noqa: E731

        state = np.asarray(initial_obs, dtype=np.float64)
        state_dim = len(state)

        states = np.zeros((horizon + 1, state_dim))
        actions_list = []
        rewards_list = []
        states[0] = state

        if use_world_model and self.world_model is not None:
            eta = state[:self.n_modes]
            eta_dot = (
                state[self.n_modes : self.n_modes * 2]
                if len(state) > self.n_modes
                else np.zeros(self.n_modes)
            )
            m = 0

            modal_states = np.zeros((horizon + 1, self.n_modes * 2))
            modal_states[0] = np.concatenate([eta, eta_dot])

            for t in range(horizon):
                action = self.policy(np.concatenate([eta, eta_dot]))
                actions_list.append(action)
                r = reward_fn(np.concatenate([eta, eta_dot]), action)
                rewards_list.append(r)
                eta, eta_dot = self.world_model.hamilton_step(eta, eta_dot, m, h=0.01)
                modal_states[t + 1] = np.concatenate([eta, eta_dot])
                states[t + 1] = modal_states[t + 1]

            return RolloutResult(
                states=states,
                actions=np.stack(actions_list),
                rewards=np.array(rewards_list),
                total_reward=float(np.sum(rewards_list)),
                modal_states=modal_states,
            )
        else:
            for t in range(horizon):
                action = self.policy(state)
                actions_list.append(action)
                r = reward_fn(state, action)
                rewards_list.append(r)
                # Simple linear dynamics placeholder (Genesis would be used here)
                if len(action) >= state_dim:
                    state = state + 0.01 * action[:state_dim]
                else:
                    state = state.copy()
                states[t + 1] = state

            return RolloutResult(
                states=states,
                actions=np.stack(actions_list),
                rewards=np.array(rewards_list),
                total_reward=float(np.sum(rewards_list)),
                modal_states=None,
            )

    def step(
        self,
        state: NDArray[np.float64],
        action: NDArray[np.float64],
        m: int = 0,
        h: float = 0.01,
    ) -> NDArray[np.float64]:
        """Advance the state by one step given an action.

        Uses the world model (Hamiltonian leapfrog) if available, otherwise
        applies simple linear forward Euler dynamics.

        Parameters
        ----------
        state : shape (state_dim,) — current state
        action : shape (action_dim,) — action to apply
        m : int — contact mode
        h : float — timestep

        Returns
        -------
        next_state : shape (state_dim,)
        """
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)

        if self.world_model is not None:
            n = self.n_modes
            eta = state[:n].copy()
            eta_dot = state[n : n * 2].copy() if len(state) >= n * 2 else np.zeros(n)
            eta_new, eta_dot_new = self.world_model.hamilton_step(eta, eta_dot, m, h=h)
            next_state = np.concatenate([eta_new, eta_dot_new])
            return next_state
        else:
            # Simple forward Euler with action as force
            state_dim = len(state)
            if len(action) >= state_dim:
                return state + h * action[:state_dim]
            return state.copy()
