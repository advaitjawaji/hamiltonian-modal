"""AMASS motion dataset retargeted to Unitree G1.

AMASS (Archive of Motion Capture as Surface Shapes) is a large collection
of MoCap data unified under the SMPL+H body model.  This loader retargets
the SMPL-format data to the G1's 23-DOF joint space.
Falls back to synthetic data when unavailable.

Reference: Mahmood et al., "AMASS: Archive of Motion Capture as Surface Shapes"
(ICCV 2019).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.data.retargeted_motions import MotionClip

logger = logging.getLogger(__name__)

G1_N_DOFS = 23
_HF_DATASET_ID = "hamiltonian-modal/amass-g1"

# AMASS sub-datasets
AMASS_SUBDATASETS = [
    "CMU", "MPI_Limits", "TotalCapture", "Eyes_Japan_Dataset",
    "KIT", "BMLmovi", "EKUT", "TCD_handMocap",
    "ACCAD", "BioMotionLab_NTroje", "SFU",
]


@dataclass
class AMASSClip(MotionClip):
    """An AMASS motion clip retargeted to G1.

    Attributes
    ----------
    subdataset : str — AMASS sub-dataset identifier
    gender : str — subject gender ("male", "female", "neutral")
    fps : float — original capture frame rate
    """
    subdataset: str = "CMU"
    gender: str = "neutral"
    fps: float = 60.0

    def __init__(
        self,
        name: str,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        dt: float = 0.01,
        subdataset: str = "CMU",
        gender: str = "neutral",
        fps: float = 60.0,
    ) -> None:
        super().__init__(name=name, q=q, qdot=qdot, dt=dt, source="amass")
        self.subdataset = subdataset
        self.gender = gender
        self.fps = fps


class AMASSSG1Dataset:
    """AMASS dataset retargeted to Unitree G1.

    Parameters
    ----------
    split : str — "train", "val", or "test"
    subdatasets : list[str] or None — filter by AMASS sub-dataset
    n_synthetic : int — fallback synthetic clips
    clip_length : int — frames per synthetic clip
    seed : int
    """

    def __init__(
        self,
        split: str = "train",
        subdatasets: "list[str] | None" = None,
        n_synthetic: int = 200,
        clip_length: int = 300,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.subdatasets = subdatasets or AMASS_SUBDATASETS
        self.n_synthetic = n_synthetic
        self.clip_length = clip_length
        self.seed = seed
        self._clips: list[AMASSClip] = []
        self._loaded = False

    def _load_from_hf(self) -> bool:
        """Attempt to load from HuggingFace."""
        try:
            from datasets import load_dataset  # type: ignore[import]
            ds = load_dataset(_HF_DATASET_ID, split=self.split)
            for i, row in enumerate(ds):
                subdataset = row.get("subdataset", "CMU")
                if subdataset not in self.subdatasets:
                    continue
                q = np.array(row["q"], dtype=np.float64)
                qdot = np.array(row["qdot"], dtype=np.float64)
                self._clips.append(AMASSClip(
                    name=row.get("name", f"amass_{i}"),
                    q=q,
                    qdot=qdot,
                    dt=float(row.get("dt", 0.01)),
                    subdataset=subdataset,
                    gender=row.get("gender", "neutral"),
                    fps=float(row.get("fps", 60.0)),
                ))
            logger.info("Loaded %d AMASS clips from HuggingFace", len(self._clips))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("AMASS HF load failed: %s. Using synthetic.", exc)
            return False

    def _generate_synthetic(self) -> None:
        """Generate synthetic clips mimicking AMASS motion diversity."""
        rng = np.random.default_rng(self.seed)
        T = self.clip_length
        dt = 0.01

        clips_per_sub = max(1, self.n_synthetic // len(self.subdatasets))
        genders = ["male", "female", "neutral"]

        for sub in self.subdatasets:
            for i in range(clips_per_sub):
                # Each sub-dataset has slightly different motion characteristics
                base_freq = rng.uniform(0.3, 2.5)
                freqs = rng.uniform(base_freq * 0.5, base_freq * 1.5, G1_N_DOFS)
                phases = rng.uniform(0.0, 2 * np.pi, G1_N_DOFS)
                amps = rng.uniform(0.05, 0.4, G1_N_DOFS)

                t = np.arange(T) * dt
                # Add second harmonic for richer motion
                q = (
                    amps[None, :] * np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])
                    + 0.3 * amps[None, :] * np.sin(4 * np.pi * freqs[None, :] * t[:, None])
                )
                qdot = np.gradient(q, dt, axis=0)

                gender = str(rng.choice(genders))
                fps = float(rng.choice([30.0, 60.0, 120.0]))

                self._clips.append(AMASSClip(
                    name=f"syn_{sub}_{i:03d}",
                    q=q.astype(np.float64),
                    qdot=qdot.astype(np.float64),
                    dt=dt,
                    subdataset=sub,
                    gender=gender,
                    fps=fps,
                ))
        logger.info("Generated %d synthetic AMASS clips", len(self._clips))

    def load(self) -> None:
        """Load the dataset."""
        if self._loaded:
            return
        if not self._load_from_hf():
            self._generate_synthetic()
        self._loaded = True

    def __len__(self) -> int:
        self.load()
        return len(self._clips)

    def __getitem__(self, idx: int) -> AMASSClip:
        self.load()
        return self._clips[idx]

    def __iter__(self):
        self.load()
        yield from self._clips

    def filter_by_subdataset(self, subdataset: str) -> list[AMASSClip]:
        """Return all clips from a given AMASS sub-dataset."""
        self.load()
        return [c for c in self._clips if c.subdataset == subdataset]

    def filter_by_gender(self, gender: str) -> list[AMASSClip]:
        """Return all clips of a given gender."""
        self.load()
        return [c for c in self._clips if c.gender == gender]
