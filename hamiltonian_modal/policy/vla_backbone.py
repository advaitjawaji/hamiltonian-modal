"""Vision-Language-Action (VLA) backbone.

Stretch goal; not implemented in v0.1.

This module is a placeholder for a future VLA backbone that would combine
a vision encoder, a language model, and an action decoder.  The interface
is defined here for forward-compatibility.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_MSG = (
    "VLABackbone is a stretch goal and is not implemented in v0.1. "
    "Use MLPPolicy or FlowMatchingHead instead."
)


class VLABackbone:
    """Vision-Language-Action backbone (stub — not implemented in v0.1).

    Parameters
    ----------
    obs_dim : int — observation dimensionality
    action_dim : int — action dimensionality
    language_dim : int — language embedding dimensionality
    vision_dim : int — vision feature dimensionality

    Notes
    -----
    Stretch goal; not implemented in v0.1.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        language_dim: int = 512,
        vision_dim: int = 512,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.language_dim = language_dim
        self.vision_dim = vision_dim
        logger.warning(_NOT_IMPLEMENTED_MSG)

    def __call__(
        self,
        obs: NDArray[np.float64],
        language_embedding: "NDArray[np.float64] | None" = None,
        vision_features: "NDArray[np.float64] | None" = None,
    ) -> NDArray[np.float64]:
        """Not implemented in v0.1.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def encode_language(self, text: str) -> NDArray[np.float64]:
        """Not implemented in v0.1."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def encode_vision(self, image: NDArray) -> NDArray[np.float64]:
        """Not implemented in v0.1."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def parameters(self) -> list[NDArray[np.float64]]:
        """Not implemented in v0.1."""
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
