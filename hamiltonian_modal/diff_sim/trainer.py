"""Training interfaces for differentiable simulation experiments.

:class:`DiffSimTrainer` wraps a :class:`~hamiltonian_modal.diff_sim.pipeline.DiffSimPipeline`
and a policy, and runs a REINFORCE-style training loop: collect episodes,
compute discounted returns, estimate parameter gradients via finite differences
over the policy loss, and apply a gradient-descent update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import DiffSimConfig
from hamiltonian_modal.diff_sim.pipeline import DiffSimPipeline, Trajectory
from hamiltonian_modal.diff_sim.policy_gradient import REINFORCEConfig, compute_returns, reinforce_update
from hamiltonian_modal.diff_sim.stability_utils import clip_grad_norm
from hamiltonian_modal.policy.mlp_policy import MLPParams, MLPPolicy

__all__ = [
    "DiffSimEpochMetrics",
    "DiffSimTrainer",
]


@dataclass
class DiffSimEpochMetrics:
    """Metrics for one training epoch."""

    epoch: int
    mean_return: float
    grad_norm: float


class DiffSimTrainer:
    """REINFORCE trainer for the differentiable simulation pipeline.

    Parameters
    ----------
    pipeline:
        :class:`~hamiltonian_modal.diff_sim.pipeline.DiffSimPipeline`.
    policy:
        :class:`~hamiltonian_modal.policy.mlp_policy.MLPPolicy`.
    sim_config:
        :class:`~hamiltonian_modal.config.DiffSimConfig`.
    pg_config:
        :class:`~hamiltonian_modal.diff_sim.policy_gradient.REINFORCEConfig`.
    """

    def __init__(
        self,
        pipeline: DiffSimPipeline,
        policy: MLPPolicy,
        sim_config: DiffSimConfig | None = None,
        pg_config: REINFORCEConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.policy = policy
        self.sim_config = sim_config or DiffSimConfig()
        self.pg_config = pg_config or REINFORCEConfig(
            lr=self.sim_config.lr,
            grad_clip=self.sim_config.grad_clip,
        )

    def _collect_trajectories(self, n: int, seed_offset: int) -> list[Trajectory]:
        trajs = []
        for i in range(n):
            traj = self.pipeline.rollout(self.policy, seed=seed_offset + i)
            trajs.append(traj)
        return trajs

    def _fd_policy_gradient(
        self, trajs: list[Trajectory], eps: float = 1e-3
    ) -> tuple[list[NDArray], list[NDArray], float]:
        """Estimate policy parameter gradients via finite differences."""
        params = self.policy.params

        def mean_return() -> float:
            ret_list = []
            for traj in trajs:
                rets = compute_returns(traj.rewards, self.pg_config.discount)
                ret_list.append(float(np.mean(rets)))
            return float(np.mean(ret_list))

        base_return = mean_return()
        w_grads, b_grads = [], []

        for j in range(len(params.weights)):
            wg = np.zeros_like(params.weights[j])
            for idx in np.ndindex(params.weights[j].shape):
                params.weights[j][idx] += eps
                r_plus = mean_return()
                params.weights[j][idx] -= 2 * eps
                r_minus = mean_return()
                params.weights[j][idx] += eps
                wg[idx] = (r_plus - r_minus) / (2 * eps)
            # Negate because we maximise return (minimise -return)
            wg = -wg
            w_grads.append(wg)

            bg = np.zeros_like(params.biases[j])
            for idx in np.ndindex(params.biases[j].shape):
                params.biases[j][idx] += eps
                r_plus = mean_return()
                params.biases[j][idx] -= 2 * eps
                r_minus = mean_return()
                params.biases[j][idx] += eps
                bg[idx] = -(r_plus - r_minus) / (2 * eps)
            b_grads.append(bg)

        return w_grads, b_grads, base_return

    def train_epoch(self, epoch: int, seed_offset: int = 0) -> DiffSimEpochMetrics:
        """Execute one training epoch.

        Parameters
        ----------
        epoch:
            Epoch index (for metrics reporting).
        seed_offset:
            Added to trajectory seeds to ensure diversity across epochs.

        Returns
        -------
        DiffSimEpochMetrics
        """
        trajs = self._collect_trajectories(self.sim_config.batch_size, seed_offset)
        w_grads, b_grads, mean_ret = self._fd_policy_gradient(trajs)

        params = self.policy.params
        all_grads = np.concatenate([g.reshape(-1) for g in w_grads + b_grads])
        all_grads, grad_norm = clip_grad_norm(all_grads, self.pg_config.grad_clip)

        lr = self.pg_config.lr
        offset = 0
        for j in range(len(params.weights)):
            n = params.weights[j].size
            params.weights[j] -= (lr * all_grads[offset: offset + n]).reshape(params.weights[j].shape)
            offset += n
        for j in range(len(params.biases)):
            n = params.biases[j].size
            params.biases[j] -= (lr * all_grads[offset: offset + n]).reshape(params.biases[j].shape)
            offset += n

        return DiffSimEpochMetrics(epoch=epoch, mean_return=mean_ret, grad_norm=grad_norm)

    def train(self, n_epochs: int | None = None) -> list[DiffSimEpochMetrics]:
        """Run the full training loop.

        Parameters
        ----------
        n_epochs:
            Number of epochs (defaults to ``sim_config.n_epochs``).

        Returns
        -------
        list[DiffSimEpochMetrics]
        """
        epochs = n_epochs or self.sim_config.n_epochs
        history = []
        for epoch in range(epochs):
            metrics = self.train_epoch(epoch, seed_offset=epoch * self.sim_config.batch_size)
            history.append(metrics)
        return history
