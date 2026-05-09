"""Per-mode quantile uncertainty heads with monotonicity enforcement.

5 quantile levels per modal coordinate: q ∈ {0.1, 0.25, 0.5, 0.75, 0.9}.
Uses pinball loss. Monotonicity enforced via softplus cumulative parameterization.
"""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
N_QUANTILES = len(QUANTILE_LEVELS)


@dataclass
class UncertaintyEstimate:
    """Output of quantile uncertainty estimation.

    Attributes
    ----------
    quantiles : NDArray, shape (n_modes, n_quantiles)
        Predicted quantiles for each mode.
    quantile_levels : tuple[float, ...]
        The quantile levels (0.1, 0.25, 0.5, 0.75, 0.9).
    mean : NDArray, shape (n_modes,)
        Median prediction (0.5 quantile).
    interval_90 : tuple[NDArray, NDArray]
        Lower and upper bounds of the 90% prediction interval.
    """
    quantiles: NDArray[np.float64]
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS

    @property
    def mean(self) -> NDArray[np.float64]:
        median_idx = list(self.quantile_levels).index(0.5)
        return self.quantiles[:, median_idx]

    @property
    def interval_90(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lo_idx = list(self.quantile_levels).index(0.1)
        hi_idx = list(self.quantile_levels).index(0.9)
        return self.quantiles[:, lo_idx], self.quantiles[:, hi_idx]


def _softplus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log1p(np.exp(np.clip(x, -50, 50)))


@dataclass
class QuantileMLPParams:
    """Parameters for a single-mode quantile MLP."""
    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        h = np.asarray(x, dtype=np.float64)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ W.T + b)
        W, b = self.weights[-1], self.biases[-1]
        raw = h @ W.T + b  # shape (N_QUANTILES,)
        # Enforce monotonicity: q1 < q2 < ... via cumulative softplus
        q = np.zeros(N_QUANTILES)
        q[0] = raw[0]
        for k in range(1, N_QUANTILES):
            q[k] = q[k - 1] + _softplus(raw[k])
        return q


class QuantileUncertainty:
    """Per-mode quantile uncertainty estimator.

    Parameters
    ----------
    n_modes : int
    n_contact_modes : int
    hidden : list[int]
    seed : int
    """

    def __init__(
        self,
        n_modes: int,
        n_contact_modes: int,
        hidden: list[int] | None = None,
        seed: int = 0,
    ) -> None:
        if hidden is None:
            hidden = [64, 64]
        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes
        self.quantile_levels = QUANTILE_LEVELS

        rng = np.random.default_rng(seed)
        n_in = n_modes + n_contact_modes
        self._mlps: list[QuantileMLPParams] = []
        for _ in range(n_modes):
            layer_sizes = [n_in] + list(hidden) + [N_QUANTILES]
            weights, biases = [], []
            for n_i, n_o in zip(layer_sizes[:-1], layer_sizes[1:]):
                lim = np.sqrt(6.0 / (n_i + n_o))
                weights.append(rng.uniform(-lim, lim, (n_o, n_i)))
                biases.append(np.zeros(n_o))
            self._mlps.append(QuantileMLPParams(weights=weights, biases=biases))

    def __call__(
        self,
        eta: NDArray[np.float64],
        m: int,
    ) -> UncertaintyEstimate:
        """Predict quantiles for each modal coordinate.

        Parameters
        ----------
        eta : shape (n_modes,)
        m : int, contact mode

        Returns
        -------
        UncertaintyEstimate
        """
        m_emb = np.zeros(self.n_contact_modes)
        m_emb[m % self.n_contact_modes] = 1.0
        x = np.concatenate([np.asarray(eta, dtype=np.float64), m_emb])

        quantiles = np.stack([mlp.forward(x) for mlp in self._mlps], axis=0)
        return UncertaintyEstimate(quantiles=quantiles)


def pinball_loss(
    y_true: NDArray[np.float64],
    quantile_preds: NDArray[np.float64],
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> float:
    """Pinball (quantile) loss.

    Parameters
    ----------
    y_true : shape (n_modes,)
    quantile_preds : shape (n_modes, n_quantiles)
    quantile_levels : tuple of quantile levels

    Returns
    -------
    loss : float (mean over modes and quantiles)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    quantile_preds = np.asarray(quantile_preds, dtype=np.float64)
    q = np.array(quantile_levels)
    error = y_true[:, None] - quantile_preds  # (n_modes, n_quantiles)
    loss = np.where(error >= 0, q[None, :] * error, (q[None, :] - 1) * error)
    return float(loss.mean())


class EnsembleUncertainty:
    """Ensemble-based uncertainty (legacy, kept for compatibility)."""

    def __init__(self, n_members: int = 5) -> None:
        self.n_members = n_members

    def predict(self, predictions: list[NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        """Compute mean and variance across ensemble members."""
        stacked = np.stack(predictions, axis=0)
        return {"mean": stacked.mean(0), "variance": stacked.var(0)}
