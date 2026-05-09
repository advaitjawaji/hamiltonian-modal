"""LAFAN1-to-G1 dataset interfaces.

Wraps the retargeted-motions loader with LAFAN1-specific conventions:

* Default frame-rate: 30 Hz → dt = 1/30 s
* Source tag: ``"LAFAN1"``
* Optional action category filtering
"""

from __future__ import annotations

from pathlib import Path

from hamiltonian_modal.data.retargeted_motions import MotionClip, load_motion_clip, load_motion_clips

__all__ = [
    "LAFAN1_DT",
    "load_lafan1_clip",
    "load_lafan1_dataset",
]

LAFAN1_DT: float = 1.0 / 30.0
"""LAFAN1 capture frame duration in seconds (30 Hz)."""


def load_lafan1_clip(path: str | Path) -> MotionClip:
    """Load a single LAFAN1-retargeted clip.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.

    Returns
    -------
    MotionClip
        Clip tagged with ``source="LAFAN1"``.
    """
    clip = load_motion_clip(path, dt=LAFAN1_DT)
    clip.source = "LAFAN1"
    return clip


def load_lafan1_dataset(
    directory: str | Path,
    actions: list[str] | None = None,
) -> list[MotionClip]:
    """Load all LAFAN1-retargeted clips from *directory*.

    Parameters
    ----------
    directory:
        Directory containing ``.npz`` archives.
    actions:
        If provided, keep only clips whose :attr:`~MotionClip.name` starts
        with one of the specified action prefixes (e.g.
        ``["walk", "run"]``).

    Returns
    -------
    list[MotionClip]
        Clips tagged with ``source="LAFAN1"``.
    """
    clips = load_motion_clips(directory, dt=LAFAN1_DT)
    for clip in clips:
        clip.source = "LAFAN1"
    if actions is not None:
        clips = [c for c in clips if any(c.name.startswith(a) for a in actions)]
    return clips
