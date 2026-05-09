"""Contact-event prediction head for the Hamiltonian world model."""
import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ContactEventPrediction:
    """Predicted contact event probabilities."""
    probabilities: NDArray[np.float64]  # shape (n_contact_modes,)
    predicted_mode: int

    @classmethod
    def from_logits(cls, logits: NDArray[np.float64]) -> "ContactEventPrediction":
        logits = np.asarray(logits, dtype=np.float64)
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        return cls(probabilities=probs, predicted_mode=int(np.argmax(probs)))


class ContactEventPredictor:
    """MLP-based contact mode prediction head.

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
            hidden = [128, 128]
        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes

        rng = np.random.default_rng(seed)
        n_in = n_modes * 2  # [η, η̇]
        layer_sizes = [n_in] + list(hidden) + [n_contact_modes]
        self._weights: list[NDArray[np.float64]] = []
        self._biases: list[NDArray[np.float64]] = []
        for n_i, n_o in zip(layer_sizes[:-1], layer_sizes[1:]):
            lim = np.sqrt(6.0 / (n_i + n_o))
            self._weights.append(rng.uniform(-lim, lim, (n_o, n_i)))
            self._biases.append(np.zeros(n_o))

    def __call__(
        self,
        eta: NDArray[np.float64],
        eta_dot: NDArray[np.float64],
    ) -> ContactEventPrediction:
        x = np.concatenate([
            np.asarray(eta, dtype=np.float64),
            np.asarray(eta_dot, dtype=np.float64),
        ])
        h = x
        for W, b in zip(self._weights[:-1], self._biases[:-1]):
            h = np.tanh(h @ W.T + b)
        logits = h @ self._weights[-1].T + self._biases[-1]
        return ContactEventPrediction.from_logits(logits)


def contact_event_loss(
    pred: ContactEventPrediction,
    true_mode: int,
) -> float:
    """Cross-entropy loss for contact event prediction."""
    log_probs = np.log(np.clip(pred.probabilities, 1e-10, 1.0))
    return float(-log_probs[true_mode])
