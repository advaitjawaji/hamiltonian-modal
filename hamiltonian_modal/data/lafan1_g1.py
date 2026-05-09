"""LAFAN1 motion dataset retargeted to Unitree G1.

LAFAN1 is a large-scale motion capture dataset of diverse human actions.
This loader retargets the original MoCap data to the G1's 23-DOF joint space.
Falls back to synthetic data when the dataset is unavailable.

Reference: Harvey et al., "Robust Motion In-Betweening" (SIGGRAPH 2020).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from hamiltonian_modal.data.retargeted_motions import MotionClip

logger = logging.getLogger(__name__)

G1_N_DOFS = 23
_HF_DATASET_ID = "hamiltonian-modal/lafan1-g1"

# LAFAN1 action categories
LAFAN1_ACTIONS = [
    "walk", "run", "jump", "dance", "carry",
    "push", "pull", "climb", "crouch", "kick",
]


@dataclass
class LAFAN1Clip(MotionClip):
    """A LAFAN1 motion clip retargeted to G1.

    Attributes
    ----------
    action : str — motion category
    subject_id : int — subject identifier (1–5)
    """
    action: str = "walk"
    subject_id: int = 1

    def __init__(
        self,
        name: str,
        q: NDArray[np.float64],
        qdot: NDArray[np.float64],
        dt: float = 0.01,
        action: str = "walk",
        subject_id: int = 1,
    ) -> None:
        super().__init__(name=name, q=q, qdot=qdot, dt=dt, source="lafan1")
        self.action = action
        self.subject_id = subject_id


class LAFAN1G1Dataset:
    """LAFAN1 dataset retargeted to Unitree G1.

    Parameters
    ----------
    split : str — "train", "val", or "test"
    actions : list[str] or None — filter by action category
    n_synthetic : int — fallback synthetic clips
    clip_length : int — frames per synthetic clip
    seed : int
    """

    def __init__(
        self,
        split: str = "train",
        actions: "list[str] | None" = None,
        n_synthetic: int = 50,
        clip_length: int = 150,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.actions = actions or LAFAN1_ACTIONS
        self.n_synthetic = n_synthetic
        self.clip_length = clip_length
        self.seed = seed
        self._clips: list[LAFAN1Clip] = []
        self._loaded = False

    def _load_from_hf(self) -> bool:
        """Attempt to load from HuggingFace."""
        try:
            from datasets import load_dataset  # type: ignore[import]
            ds = load_dataset(_HF_DATASET_ID, split=self.split)
            for i, row in enumerate(ds):
                action = row.get("action", "walk")
                if action not in self.actions:
                    continue
                q = np.array(row["q"], dtype=np.float64)
                qdot = np.array(row["qdot"], dtype=np.float64)
                self._clips.append(LAFAN1Clip(
                    name=row.get("name", f"lafan1_{i}"),
                    q=q,
                    qdot=qdot,
                    dt=float(row.get("dt", 0.01)),
                    action=action,
                    subject_id=int(row.get("subject_id", 1)),
                ))
            logger.info("Loaded %d LAFAN1 clips", len(self._clips))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("LAFAN1 HF load failed: %s. Using synthetic.", exc)
            return False

    def _generate_synthetic(self) -> None:
        """Generate synthetic clips labelled with LAFAN1 action categories."""
        rng = np.random.default_rng(self.seed)
        T = self.clip_length
        dt = 0.01

        action_params = {
            "walk": (0.5, 0.1),
            "run": (2.0, 0.2),
            "jump": (3.0, 0.25),
            "dance": (1.5, 0.15),
            "carry": (0.3, 0.08),
            "push": (0.4, 0.09),
            "pull": (0.4, 0.09),
            "climb": (0.6, 0.12),
            "crouch": (0.2, 0.07),
            "kick": (2.5, 0.22),
        }

        clips_per_action = max(1, self.n_synthetic // len(self.actions))
        for action in self.actions:
            freq_scale, amp_scale = action_params.get(action, (1.0, 0.1))
            for i in range(clips_per_action):
                freqs = rng.uniform(freq_scale * 0.8, freq_scale * 1.2, G1_N_DOFS)
                phases = rng.uniform(0.0, 2 * np.pi, G1_N_DOFS)
                amps = rng.uniform(amp_scale * 0.8, amp_scale * 1.2, G1_N_DOFS)

                t = np.arange(T) * dt
                q = amps[None, :] * np.sin(
                    2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]
                )
                qdot = (amps * 2 * np.pi * freqs)[None, :] * np.cos(
                    2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]
                )

                subject_id = int(rng.integers(1, 6))
                self._clips.append(LAFAN1Clip(
                    name=f"syn_{action}_{i:03d}",
                    q=q.astype(np.float64),
                    qdot=qdot.astype(np.float64),
                    dt=dt,
                    action=action,
                    subject_id=subject_id,
                ))
        logger.info("Generated %d synthetic LAFAN1 clips", len(self._clips))

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

    def __getitem__(self, idx: int) -> LAFAN1Clip:
        self.load()
        return self._clips[idx]

    def __iter__(self):
        self.load()
        yield from self._clips

    def filter_by_action(self, action: str) -> list[LAFAN1Clip]:
        """Return all clips of a given action category."""
        self.load()
        return [c for c in self._clips if c.action == action]
