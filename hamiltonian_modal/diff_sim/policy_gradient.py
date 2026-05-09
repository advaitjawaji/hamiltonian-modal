"""Policy-gradient interfaces for differentiable simulation.

Implements REINFORCE (Williams 1992) with optional baseline subtraction.
Because the simulation loop is built in NumPy (no autograd), parameter
gradients are estimated via REINFORCE rather than backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.diff_sim.stability_utils import clip_grad_norm, stable_log

__all__ = [
    "REINFORCEConfig",
    "compute_returns",
    "reinforce_update",
]


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE policy gradient.

    Attributes
    ----------
    discount:
        Discount factor γ.
    use_baseline:
        Subtract mean return as baseline.
    lr:
        Learning rate for parameter update.
    grad_clip:
        Maximum gradient norm (0 = disabled).
    """

    discount: float = 0.99
    use_baseline: bool = True
    lr: float = 1e-3
    grad_clip: float = 1.0


def compute_returns(
    rewards: NDArray[np.float64],
    discount: float = 0.99,
) -> NDArray[np.float64]:
    """Compute discounted returns G_t for each timestep.

    Parameters
    ----------
    rewards:
        Per-step reward array, shape ``(T,)``.
    discount:
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    NDArray[np.float64]
        Discounted returns, shape ``(T,)``.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + discount * G
        returns[t] = G
    return returns


def reinforce_update(
    log_probs: NDArray[np.float64],
    returns: NDArray[np.float64],
    config: REINFORCEConfig | None = None,
) -> tuple[float, float]:
    """Compute the REINFORCE policy-gradient loss and grad norm.

    The scalar loss is

    .. math::

        \\mathcal{L} = -\\frac{1}{T} \\sum_t (G_t - b) \\log \\pi(a_t | s_t)

    where b is the mean return (when ``use_baseline=True``).

    Parameters
    ----------
    log_probs:
        Log-probabilities log π(aₜ | sₜ), shape ``(T,)``.
    returns:
        Discounted returns G_t, shape ``(T,)``.
    config:
        :class:`REINFORCEConfig`.

    Returns
    -------
    tuple[float, float]
        ``(loss, grad_norm)`` where grad_norm is the L2 norm of the loss
        gradient w.r.t. log_probs (useful for monitoring).
    """
    cfg = config or REINFORCEConfig()
    baseline = np.mean(returns) if cfg.use_baseline else 0.0
    advantages = returns - baseline
    # Loss: negative expected return
    loss = -float(np.mean(advantages * log_probs))
    # Gradient of loss w.r.t. log_probs (for monitoring)
    grad = -advantages / len(advantages)
    grad, grad_norm = clip_grad_norm(grad, cfg.grad_clip)
    return loss, grad_norm
