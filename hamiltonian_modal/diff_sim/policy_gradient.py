"""Policy gradient via differentiable simulation.

Implements exact backprop through the diff-sim pipeline when JAX is available,
or finite-difference approximation otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Optional JAX import
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    logger.debug("JAX not available; using finite-difference policy gradients")


# ---------------------------------------------------------------------------
# REINFORCE (comparison baseline)
# ---------------------------------------------------------------------------

@dataclass
class REINFORCEConfig:
    """Configuration for the REINFORCE algorithm.

    Parameters
    ----------
    gamma : float — discount factor
    learning_rate : float — policy gradient step size
    baseline : str — "none" or "mean" variance-reduction baseline
    n_rollouts : int — number of rollouts per gradient estimate
    """
    gamma: float = 0.99
    learning_rate: float = 1e-3
    baseline: str = "mean"
    n_rollouts: int = 16


def compute_returns(
    rewards: NDArray[np.float64],
    gamma: float = 0.99,
) -> NDArray[np.float64]:
    """Compute discounted returns G_t = Σ_{k>=t} γ^(k-t) r_k.

    Parameters
    ----------
    rewards : shape (T,) — per-step rewards
    gamma : float — discount factor

    Returns
    -------
    returns : shape (T,) — discounted cumulative returns
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def reinforce_update(
    log_probs: NDArray[np.float64],
    returns: NDArray[np.float64],
    baseline: str = "mean",
) -> NDArray[np.float64]:
    """Compute the REINFORCE policy gradient loss (for backprop).

    L = -Σ_t log π(a_t|s_t) · (G_t - b)

    Parameters
    ----------
    log_probs : shape (T,) — log probabilities of taken actions
    returns : shape (T,) — discounted returns from compute_returns
    baseline : str — "none" or "mean"

    Returns
    -------
    losses : shape (T,) — per-timestep policy gradient losses
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)

    if baseline == "mean":
        advantages = returns - np.mean(returns)
    elif baseline == "none":
        advantages = returns.copy()
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    # Normalise advantages for stability
    std = float(np.std(advantages))
    if std > 1e-8:
        advantages = advantages / std

    return -log_probs * advantages


# ---------------------------------------------------------------------------
# Diff-sim policy gradient
# ---------------------------------------------------------------------------

def diff_sim_policy_gradient(
    pipeline: Any,
    initial_states: NDArray[np.float64],
    horizon: int,
    reward_fn: Callable[[NDArray[np.float64]], float],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Compute the policy gradient via backprop through the diff-sim pipeline.

    When JAX is available, uses jax.grad for exact gradients.
    Otherwise falls back to central finite differences w.r.t. each policy
    parameter.

    Parameters
    ----------
    pipeline : DiffSimPipeline — differentiable simulation pipeline
    initial_states : shape (batch, state_dim) — starting states
    horizon : int — rollout length
    reward_fn : callable — state → reward scalar
    eps : float — finite-difference perturbation size

    Returns
    -------
    grad : NDArray matching the flattened policy parameter vector
    """
    params = pipeline.policy.parameters()

    if _JAX_AVAILABLE:
        try:
            return _jax_policy_gradient(pipeline, initial_states, horizon, reward_fn, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JAX gradient failed (%s); falling back to FD", exc)

    return _fd_policy_gradient(pipeline, initial_states, horizon, reward_fn, params, eps)


def _rollout_reward(
    pipeline: Any,
    initial_states: NDArray[np.float64],
    horizon: int,
    reward_fn: Callable[[NDArray[np.float64]], float],
) -> float:
    """Execute a pipeline rollout and sum rewards."""
    batch_size = initial_states.shape[0]
    total_reward = 0.0
    for b in range(batch_size):
        state = initial_states[b].copy()
        for _ in range(horizon):
            action = pipeline.policy(state)
            state = pipeline.step(state, action)
            total_reward += reward_fn(state)
    return total_reward / batch_size


def _fd_policy_gradient(
    pipeline: Any,
    initial_states: NDArray[np.float64],
    horizon: int,
    reward_fn: Callable[[NDArray[np.float64]], float],
    params: list[NDArray[np.float64]],
    eps: float,
) -> NDArray[np.float64]:
    """Central finite-difference policy gradient estimate."""
    flat_params = np.concatenate([p.ravel() for p in params])
    n_params = len(flat_params)
    grad = np.zeros(n_params, dtype=np.float64)

    # Pointer list to map flat index back to parameter tensors
    param_sizes = [p.size for p in params]
    param_shapes = [p.shape for p in params]

    def set_flat_params(flat: NDArray[np.float64]) -> None:
        offset = 0
        new_params = []
        for size, shape in zip(param_sizes, param_shapes):
            new_params.append(flat[offset : offset + size].reshape(shape).copy())
            offset += size
        pipeline.policy.set_parameters(new_params)

    for i in range(n_params):
        params_p = flat_params.copy()
        params_p[i] += eps
        set_flat_params(params_p)
        r_p = _rollout_reward(pipeline, initial_states, horizon, reward_fn)

        params_m = flat_params.copy()
        params_m[i] -= eps
        set_flat_params(params_m)
        r_m = _rollout_reward(pipeline, initial_states, horizon, reward_fn)

        grad[i] = (r_p - r_m) / (2.0 * eps)

    # Restore original parameters
    set_flat_params(flat_params)
    return grad


def _jax_policy_gradient(
    pipeline: Any,
    initial_states: NDArray[np.float64],
    horizon: int,
    reward_fn: Callable[[NDArray[np.float64]], float],
    params: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """JAX-based exact policy gradient via jax.grad."""
    flat_params = jnp.array(np.concatenate([p.ravel() for p in params]))
    param_sizes = [p.size for p in params]
    param_shapes = [p.shape for p in params]

    def total_reward_jax(flat: Any) -> Any:
        # Unpack flat params → update policy → rollout → reward
        offset = 0
        new_params = []
        for size, shape in zip(param_sizes, param_shapes):
            new_params.append(flat[offset : offset + size].reshape(shape))
            offset += size
        pipeline.policy.set_parameters([np.array(p) for p in new_params])
        r = _rollout_reward(pipeline, initial_states, horizon, reward_fn)
        return jnp.array(r)

    grad_fn = jax.grad(total_reward_jax)
    grad_jax = grad_fn(flat_params)
    return np.array(grad_jax)
