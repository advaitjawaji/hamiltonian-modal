"""VLA backbone interfaces.

Thin adapter layer that wraps a Vision-Language-Action (VLA) model backbone
and exposes a uniform ``encode(obs) → feature_vector`` interface.

Because VLA models are large multi-modal models that depend on specific
frameworks (e.g. PyTorch / JAX), the backbone is **lazy-loaded**: the actual
model is passed in at construction time so the rest of the package remains
framework-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "VLABackbone",
]


class VLABackbone:
    """Framework-agnostic adapter for Vision-Language-Action backbones.

    Parameters
    ----------
    model:
        Any callable ``model(obs_dict) → feature_tensor`` (e.g. a PyTorch
        ``nn.Module`` or a JAX function).  The output is converted to a
        NumPy array before being returned.
    feature_dim:
        Dimensionality of the feature vector emitted by *model*.
    preprocessor:
        Optional callable ``obs_dict → obs_dict`` applied before *model*.
    postprocessor:
        Optional callable ``raw_output → NDArray`` that converts the model
        output to a 1-D NumPy float64 array.  Defaults to
        ``np.asarray(x).reshape(-1)``.
    """

    def __init__(
        self,
        model: Any,
        feature_dim: int,
        preprocessor: Callable | None = None,
        postprocessor: Callable | None = None,
    ) -> None:
        self._model = model
        self.feature_dim = feature_dim
        self._pre = preprocessor
        self._post = postprocessor or (lambda x: np.asarray(x, dtype=np.float64).reshape(-1))

    def encode(self, obs: Any) -> NDArray[np.float64]:
        """Encode *obs* into a feature vector.

        Parameters
        ----------
        obs:
            Raw observation (dict, tensor, or numpy array depending on the
            framework).

        Returns
        -------
        NDArray[np.float64]
            Feature vector, shape ``(feature_dim,)``.
        """
        if self._pre is not None:
            obs = self._pre(obs)
        raw = self._model(obs)
        features = self._post(raw)
        if features.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected feature vector of shape ({self.feature_dim},); "
                f"got {features.shape}."
            )
        return features
