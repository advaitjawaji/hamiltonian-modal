"""Retargeted motion dataset interfaces.

Provides :class:`MotionClip` — a lightweight container for a single retargeted
motion sequence — and :func:`load_motion_clip` / :func:`load_motion_clips` for
reading NumPy ``.npz`` archives produced by any retargeting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MotionClip",
    "load_motion_clip",
    "load_motion_clips",
]


@dataclass
class MotionClip:
    """A single retargeted motion sequence.

    Attributes
    ----------
    name:
        Clip identifier (file stem or user-supplied label).
    q:
        Joint configurations, shape ``(T, n_dof)``.
    v:
        Joint velocities, shape ``(T, n_dof)``.
    dt:
        Timestep in seconds between consecutive frames.
    source:
        Optional tag indicating the source dataset (e.g. ``"LAFAN1"``).
    """

    name: str
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    dt: float
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of frames in the clip."""
        return self.q.shape[0]

    @property
    def n_dof(self) -> int:
        """Degrees of freedom."""
        return self.q.shape[1]

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.n_frames * self.dt


def load_motion_clip(path: str | Path, dt: float = 0.033) -> MotionClip:
    """Load a single motion clip from an ``.npz`` archive.

    The archive must contain at least the key ``"q"`` (joint positions) of
    shape ``(T, n_dof)``.  A ``"v"`` key is optional; if absent, velocities
    are estimated via finite differences.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.
    dt:
        Frame duration in seconds (used when ``"dt"`` is absent from the
        archive).

    Returns
    -------
    MotionClip
    """
    p = Path(path)
    data = np.load(str(p))
    q = np.asarray(data["q"], dtype=np.float64)
    if "v" in data:
        v = np.asarray(data["v"], dtype=np.float64)
    else:
        # Central-difference finite-difference estimate
        v = np.zeros_like(q)
        effective_dt = float(data["dt"]) if "dt" in data else dt
        v[1:-1] = (q[2:] - q[:-2]) / (2.0 * effective_dt)
        v[0] = v[1]
        v[-1] = v[-2]
    resolved_dt = float(data["dt"]) if "dt" in data else dt
    extra = {k: data[k] for k in data.files if k not in ("q", "v", "dt")}
    return MotionClip(name=p.stem, q=q, v=v, dt=resolved_dt, extra=extra)


def load_motion_clips(directory: str | Path, dt: float = 0.033) -> list[MotionClip]:
    """Load all ``.npz`` motion clips from *directory*.

    Parameters
    ----------
    directory:
        Directory containing ``.npz`` clip archives.
    dt:
        Default frame duration in seconds (used when absent from an archive).

    Returns
    -------
    list[MotionClip]
        Clips sorted by file name.
    """
    d = Path(directory)
    clips = []
    for npz_path in sorted(d.glob("*.npz")):
        clips.append(load_motion_clip(npz_path, dt=dt))
    return clips
