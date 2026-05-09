"""Training-loop interfaces for Hamiltonian world models.

Provides :class:`WorldModelTrainer` — a self-contained training loop that
fits a :class:`~hamiltonian_modal.world_model.hamiltonian_net.HamiltonianNet`
to (η, μ, H_target) tuples using a simple finite-difference gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams, finite_diff_grad

__all__ = [
    "TrainingBatch",
    "TrainingMetrics",
    "WorldModelTrainer",
]


@dataclass
class TrainingBatch:
    """A mini-batch of modal-space training data.

    Attributes
    ----------
    eta:
        Modal positions, shape ``(B, n_modes)``.
    mu:
        Modal momenta, shape ``(B, n_modes)``.
    H_target:
        Target Hamiltonian values, shape ``(B,)``.
    """

    eta: NDArray[np.float64]
    mu: NDArray[np.float64]
    H_target: NDArray[np.float64]

    @property
    def batch_size(self) -> int:
        return self.eta.shape[0]


@dataclass
class TrainingMetrics:
    """Metrics for one training epoch."""

    epoch: int
    loss: float
    """Mean squared error between predicted and target Hamiltonian values."""


class WorldModelTrainer:
    """Gradient-descent trainer for :class:`HamiltonianNet`.

    Optimises the MSE loss

    .. math::

        \\mathcal{L} = \\frac{1}{B} \\sum_{i} (H_{\\theta}(\\eta_i, \\mu_i) - H_i^*)^2

    using finite-difference gradients w.r.t. each scalar parameter.

    Parameters
    ----------
    net:
        The Hamiltonian network to train.
    lr:
        Learning rate.
    fd_eps:
        Finite-difference step for parameter gradients.
    """

    def __init__(
        self,
        net: HamiltonianNet,
        lr: float = 1e-3,
        fd_eps: float = 1e-4,
    ) -> None:
        self.net = net
        self.lr = lr
        self.fd_eps = fd_eps

    def _mse_loss(self, batch: TrainingBatch) -> float:
        preds = np.array(
            [self.net(batch.eta[i], batch.mu[i]) for i in range(batch.batch_size)],
            dtype=np.float64,
        )
        return float(np.mean((preds - batch.H_target) ** 2))

    def train_step(self, batch: TrainingBatch) -> float:
        """Perform one gradient-descent step on *batch*.

        Parameters
        ----------
        batch:
            Training mini-batch.

        Returns
        -------
        float
            MSE loss before the update.
        """
        loss_before = self._mse_loss(batch)
        params = self.net.params

        # Update W_kinetic
        def loss_kinetic(flat: NDArray[np.float64]) -> float:
            params.W_kinetic = flat.reshape(params.W_kinetic.shape)
            return self._mse_loss(batch)

        flat_k = params.W_kinetic.reshape(-1)
        grad_k = finite_diff_grad(loss_kinetic, flat_k, eps=self.fd_eps)
        params.W_kinetic = (flat_k - self.lr * grad_k).reshape(params.W_kinetic.shape)

        # Update MLP weights
        for j in range(len(params.V_weights)):
            w_flat = params.V_weights[j].reshape(-1)

            def loss_w(flat: NDArray[np.float64], j: int = j) -> float:
                params.V_weights[j] = flat.reshape(params.V_weights[j].shape)
                return self._mse_loss(batch)

            grad_w = finite_diff_grad(loss_w, w_flat, eps=self.fd_eps)
            params.V_weights[j] = (w_flat - self.lr * grad_w).reshape(params.V_weights[j].shape)

            b_vec = params.V_biases[j]

            def loss_b(flat: NDArray[np.float64], j: int = j) -> float:
                params.V_biases[j] = flat
                return self._mse_loss(batch)

            grad_b = finite_diff_grad(loss_b, b_vec, eps=self.fd_eps)
            params.V_biases[j] = b_vec - self.lr * grad_b

        return loss_before

    def train(
        self, batches: list[TrainingBatch], n_epochs: int = 10
    ) -> list[TrainingMetrics]:
        """Run the training loop for *n_epochs*.

        Parameters
        ----------
        batches:
            List of mini-batches (iterated in order each epoch).
        n_epochs:
            Number of passes over *batches*.

        Returns
        -------
        list[TrainingMetrics]
        """
        history: list[TrainingMetrics] = []
        for epoch in range(n_epochs):
            epoch_losses = [self.train_step(b) for b in batches]
            metrics = TrainingMetrics(epoch=epoch, loss=float(np.mean(epoch_losses)))
            history.append(metrics)
        return history
