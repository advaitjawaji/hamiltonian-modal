"""End-to-end differentiable simulation trainer."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.diff_sim.stability_utils import (
    clip_grad_norm,
    trajectory_length_curriculum,
    check_grad_health,
)
from hamiltonian_modal.diff_sim.policy_gradient import diff_sim_policy_gradient

logger = logging.getLogger(__name__)


@dataclass
class DiffSimTrainConfig:
    """Configuration for the differentiable-simulation trainer.

    Parameters
    ----------
    n_iterations : int — total gradient steps
    batch_size : int — number of parallel rollouts per gradient step
    learning_rate : float — Adam learning rate
    grad_clip : float — global gradient norm clip threshold
    horizon_start : int — curriculum initial horizon
    horizon_end : int — curriculum final horizon
    seed : int — global random seed
    log_interval : int — steps between log messages
    adam_beta1 : float
    adam_beta2 : float
    adam_eps : float
    """
    n_iterations: int = 10_000
    batch_size: int = 16
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    horizon_start: int = 10
    horizon_end: int = 100
    seed: int = 42
    log_interval: int = 100
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class _AdamOptimiser:
    """Simple Adam optimiser operating on a flat parameter vector."""

    def __init__(self, lr: float, beta1: float, beta2: float, eps: float, n_params: int) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.t = 0

    def step(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute parameter update and return delta (to subtract from params)."""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad ** 2
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class DiffSimTrainer:
    """End-to-end differentiable simulation trainer.

    Trains the policy in ``pipeline`` by rolling out trajectories in the
    differentiable simulator and back-propagating through the physics to
    compute policy gradients.  Gradient clipping and a horizon curriculum
    are applied for stability.

    Parameters
    ----------
    config : DiffSimTrainConfig
    pipeline : DiffSimPipeline — wraps policy + world model
    """

    def __init__(
        self,
        config: DiffSimTrainConfig,
        pipeline: Any,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self._rng = np.random.default_rng(config.seed)
        self._iteration = 0

        # Initialise Adam on the flattened parameter vector
        params = pipeline.policy.parameters()
        n_params = sum(p.size for p in params)
        self._optimiser = _AdamOptimiser(
            lr=config.learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
            n_params=n_params,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        env_fn: Callable[[], Any],
        log_to_wandb: bool = False,
    ) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        env_fn : callable — factory that returns a new environment instance
        log_to_wandb : bool — if True, attempt to log metrics to wandb

        Returns
        -------
        history : dict with keys "rewards", "grad_norms", "horizons"
        """
        history: dict[str, list[float]] = {
            "rewards": [],
            "grad_norms": [],
            "horizons": [],
        }

        cfg = self.config
        t0 = time.monotonic()

        for step in range(cfg.n_iterations):
            horizon = trajectory_length_curriculum(
                step, cfg.n_iterations, cfg.horizon_start, cfg.horizon_end
            )

            # Sample initial states from the environment
            env = env_fn()
            initial_states = np.stack([
                env.reset() for _ in range(cfg.batch_size)
            ])

            # Define reward function (uses the env's reward logic)
            def reward_fn(state: NDArray[np.float64]) -> float:
                env2 = env_fn()
                env2._state = state.copy()
                return env2._compute_reward(np.zeros(env2.action_dim))

            # Compute policy gradient
            grad_flat = diff_sim_policy_gradient(
                self.pipeline,
                initial_states,
                horizon,
                reward_fn,
                eps=1e-4,
            )

            # Clip gradient
            params = self.pipeline.policy.parameters()
            param_sizes = [p.size for p in params]
            grad_list = []
            offset = 0
            for size, p in zip(param_sizes, params):
                grad_list.append(grad_flat[offset : offset + size].reshape(p.shape).copy())
                offset += size

            clipped_grads, grad_norm = clip_grad_norm(grad_list, cfg.grad_clip)

            # Adam update
            clipped_flat = np.concatenate([g.ravel() for g in clipped_grads])
            delta = self._optimiser.step(clipped_flat)

            # Apply update
            flat_params = np.concatenate([p.ravel() for p in params])
            new_flat = flat_params + delta  # grad is reward gradient, add to maximise
            new_params = []
            offset = 0
            for size, shape in zip(param_sizes, [p.shape for p in params]):
                new_params.append(new_flat[offset : offset + size].reshape(shape).copy())
                offset += size
            self.pipeline.policy.set_parameters(new_params)

            # Compute mean reward for logging
            mean_reward = float(np.mean([
                sum(
                    reward_fn(self.pipeline.step(initial_states[b], self.pipeline.policy(initial_states[b])))
                    for _ in range(min(5, horizon))
                )
                for b in range(min(4, cfg.batch_size))
            ]))

            history["rewards"].append(mean_reward)
            history["grad_norms"].append(grad_norm)
            history["horizons"].append(float(horizon))

            if step % cfg.log_interval == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Step %d/%d | horizon=%d | reward=%.4f | grad_norm=%.4f | elapsed=%.1fs",
                    step,
                    cfg.n_iterations,
                    horizon,
                    mean_reward,
                    grad_norm,
                    elapsed,
                )

            if log_to_wandb:
                try:
                    import wandb
                    wandb.log({
                        "reward": mean_reward,
                        "grad_norm": grad_norm,
                        "horizon": horizon,
                    }, step=step)
                except ImportError:
                    pass

            self._iteration = step + 1

        return history
