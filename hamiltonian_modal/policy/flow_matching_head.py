"""Conditional flow matching policy head.

Implements a simple conditional flow matching (CFM) policy head that
generates actions by integrating a learned vector field from noise to
a target action distribution.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2022).
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0.0)


def _tanh(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.tanh(x)


class FlowMatchingHead:
    """Conditional flow matching policy head.

    Learns a time-dependent velocity field u_θ(x, t, c) that transports
    samples from a standard Gaussian prior to the action distribution
    conditioned on the context c (e.g. observation or latent state).

    The velocity field is parameterised as a small MLP that takes
    [x; t; c] as input and outputs the velocity in action space.

    Parameters
    ----------
    action_dim : int — dimensionality of the action space
    context_dim : int — dimensionality of the conditioning context
    hidden : list[int] — hidden layer sizes for the velocity MLP
    n_steps : int — number of Euler integration steps during inference
    seed : int
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        hidden: "list[int] | None" = None,
        n_steps: int = 10,
        seed: int = 0,
    ) -> None:
        if hidden is None:
            hidden = [128, 128]

        self.action_dim = action_dim
        self.context_dim = context_dim
        self.n_steps = n_steps

        # Input: [x (action_dim), t (1), context (context_dim)]
        in_dim = action_dim + 1 + context_dim

        rng = np.random.default_rng(seed)
        layer_sizes = [in_dim] + list(hidden) + [action_dim]

        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []

        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(2.0 / n_in)
            W = rng.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
            b = np.zeros(n_out, dtype=np.float64)
            self._weights.append(W)
            self._biases.append(b)

        self._rng = np.random.default_rng(seed + 1)

    # ------------------------------------------------------------------
    # Velocity field
    # ------------------------------------------------------------------

    def velocity(
        self,
        x: NDArray[np.float64],
        t: float,
        context: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate the learned velocity field u_θ(x, t, context).

        Parameters
        ----------
        x : shape (action_dim,) — current point in action space
        t : float in [0, 1] — flow time
        context : shape (context_dim,) — conditioning information

        Returns
        -------
        v : shape (action_dim,) — velocity vector
        """
        t_arr = np.array([t], dtype=np.float64)
        h = np.concatenate([
            np.asarray(x, dtype=np.float64),
            t_arr,
            np.asarray(context, dtype=np.float64),
        ])
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            h = _tanh(h @ W.T + b)
        W, b = self._weights[-1], self._biases[-1]
        return h @ W.T + b

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def sample(
        self,
        context: NDArray[np.float64],
        rng: "np.random.Generator | None" = None,
    ) -> NDArray[np.float64]:
        """Generate an action by integrating the velocity field from t=0 to t=1.

        Starts from a standard Gaussian sample and applies Euler integration.

        Parameters
        ----------
        context : shape (context_dim,)
        rng : numpy Generator for the initial noise sample

        Returns
        -------
        action : shape (action_dim,)
        """
        if rng is None:
            rng = self._rng

        x = rng.standard_normal(self.action_dim).astype(np.float64)
        dt = 1.0 / self.n_steps
        context = np.asarray(context, dtype=np.float64)

        for i in range(self.n_steps):
            t = i * dt
            v = self.velocity(x, t, context)
            x = x + dt * v

        return np.tanh(x)  # squash to [-1, 1]

    def __call__(
        self,
        context: NDArray[np.float64],
        rng: "np.random.Generator | None" = None,
    ) -> NDArray[np.float64]:
        """Alias for sample()."""
        return self.sample(context, rng)

    # ------------------------------------------------------------------
    # CFM training objective
    # ------------------------------------------------------------------

    def cfm_loss(
        self,
        x_0: NDArray[np.float64],
        x_1: NDArray[np.float64],
        context: NDArray[np.float64],
        t: float,
    ) -> float:
        """Conditional flow matching loss at time t.

        The target velocity is the straight-line field:
            u_t(x | x_0, x_1) = x_1 - x_0

        L = ||u_θ(x_t, t, c) - (x_1 - x_0)||²

        Parameters
        ----------
        x_0 : shape (action_dim,) — source (noise) sample
        x_1 : shape (action_dim,) — target (data) sample
        context : shape (context_dim,)
        t : float in (0, 1)

        Returns
        -------
        loss : float
        """
        x_0 = np.asarray(x_0, dtype=np.float64)
        x_1 = np.asarray(x_1, dtype=np.float64)
        x_t = (1.0 - t) * x_0 + t * x_1  # linear interpolant
        target_v = x_1 - x_0
        pred_v = self.velocity(x_t, t, context)
        residual = pred_v - target_v
        return float(0.5 * np.dot(residual, residual))

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def parameters(self) -> list[NDArray[np.float64]]:
        """Return all trainable parameters in [W_0, b_0, ...] order."""
        params: list[NDArray[np.float64]] = []
        for W, b in zip(self._weights, self._biases):
            params.append(W)
            params.append(b)
        return params

    def set_parameters(self, params: list[NDArray[np.float64]]) -> None:
        """Set parameters from a list in the same order as parameters()."""
        n_layers = len(self._weights)
        if len(params) != 2 * n_layers:
            raise ValueError(
                f"Expected {2 * n_layers} parameter arrays, got {len(params)}"
            )
        for i in range(n_layers):
            self._weights[i] = np.asarray(params[2 * i], dtype=np.float64).copy()
            self._biases[i] = np.asarray(params[2 * i + 1], dtype=np.float64).copy()
