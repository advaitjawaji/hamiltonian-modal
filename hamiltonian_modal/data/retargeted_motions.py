"""Retargeted motion dataset loader.

Loads retargeted human motion data mapped to the Unitree G1 joint space.
Falls back to synthetic data when HuggingFace datasets are unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

G1_N_DOFS = 23
_HF_DATASET_ID = "hamiltonian-modal/retargeted-motions-g1"


@dataclass
class MotionClip:
    """A single retargeted motion clip.

    Attributes
    ----------
    name : str — clip identifier
    q : shape (T, n_dofs) — joint positions
    qdot : shape (T, n_dofs) — joint velocities
    dt : float — timestep between frames
    source : str — source dataset ("lafan1", "amass", "synthetic")
    """
    name: str
    q: NDArray[np.float64]
    qdot: NDArray[np.float64]
    dt: float = 0.01
    source: str = "synthetic"

    @property
    def n_frames(self) -> int:
        """Number of frames in the clip."""
        return self.q.shape[0]

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.n_frames * self.dt


class RetargetedMotionDataset:
    """Dataset of retargeted motions for the Unitree G1.

    Tries to load from HuggingFace; falls back to synthetic data.

    Parameters
    ----------
    split : str — "train", "val", or "test"
    n_synthetic : int — number of synthetic clips to generate if real data unavailable
    clip_length : int — frames per synthetic clip
    seed : int — random seed for synthetic generation
    """

    def __init__(
        self,
        split: str = "train",
        n_synthetic: int = 100,
        clip_length: int = 200,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.n_synthetic = n_synthetic
        self.clip_length = clip_length
        self.seed = seed
        self._clips: list[MotionClip] = []
        self._loaded = False

    def _load_from_hf(self) -> bool:
        """Attempt to load from HuggingFace datasets."""
        try:
            from datasets import load_dataset  # type: ignore[import]
            ds = load_dataset(_HF_DATASET_ID, split=self.split)
            for i, row in enumerate(ds):
                q = np.array(row["q"], dtype=np.float64)
                qdot = np.array(row["qdot"], dtype=np.float64)
                self._clips.append(MotionClip(
                    name=row.get("name", f"clip_{i}"),
                    q=q,
                    qdot=qdot,
                    dt=float(row.get("dt", 0.01)),
                    source=row.get("source", "retargeted"),
                ))
            logger.info("Loaded %d clips from HuggingFace (%s)", len(self._clips), self.split)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load from HuggingFace: %s. Using synthetic data.", exc)
            return False

    def _generate_synthetic(self) -> None:
        """Generate synthetic sinusoidal motion clips."""
        rng = np.random.default_rng(self.seed)
        T = self.clip_length
        dt = 0.01

        for i in range(self.n_synthetic):
            # Random sinusoidal motion per DOF
            freqs = rng.uniform(0.5, 3.0, G1_N_DOFS)
            phases = rng.uniform(0.0, 2 * np.pi, G1_N_DOFS)
            amps = rng.uniform(0.05, 0.3, G1_N_DOFS)

            t = np.arange(T, dtype=np.float64) * dt
            q = amps[None, :] * np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])
            qdot = (amps * 2 * np.pi * freqs)[None, :] * np.cos(
                2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]
            )

            self._clips.append(MotionClip(
                name=f"synthetic_{self.split}_{i:04d}",
                q=q,
                qdot=qdot,
                dt=dt,
                source="synthetic",
            ))
        logger.info("Generated %d synthetic clips", self.n_synthetic)

    def load(self) -> None:
        """Load or generate the dataset."""
        if self._loaded:
            return
        success = self._load_from_hf()
        if not success:
            self._generate_synthetic()
        self._loaded = True

    def __len__(self) -> int:
        self.load()
        return len(self._clips)

    def __getitem__(self, idx: int) -> MotionClip:
        self.load()
        return self._clips[idx]

    def __iter__(self):
        self.load()
        yield from self._clips

    def get_batch(
        self,
        batch_size: int,
        rng: "np.random.Generator | None" = None,
    ) -> list[MotionClip]:
        """Sample a random batch of clips.

        Parameters
        ----------
        batch_size : int
        rng : numpy Generator

        Returns
        -------
        batch : list[MotionClip]
        """
        self.load()
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(len(self._clips), size=min(batch_size, len(self._clips)), replace=False)
        return [self._clips[i] for i in idx]
