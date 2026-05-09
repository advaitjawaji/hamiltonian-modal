"""AMASS-to-G1 dataset interfaces.

Wraps the retargeted-motions loader with AMASS-specific conventions:

* Default frame-rate: 60 Hz → dt = 1/60 s
* Source tag: ``"AMASS"``
"""

from __future__ import annotations

from pathlib import Path

from hamiltonian_modal.data.retargeted_motions import MotionClip, load_motion_clip, load_motion_clips

__all__ = [
    "AMASS_DT",
    "load_amass_clip",
    "load_amass_dataset",
]

AMASS_DT: float = 1.0 / 60.0
"""AMASS capture frame duration in seconds (60 Hz)."""


def load_amass_clip(path: str | Path) -> MotionClip:
    """Load a single AMASS-retargeted clip.

    Parameters
    ----------
    path:
        Path to the ``.npz`` archive.

    Returns
    -------
    MotionClip
        Clip tagged with ``source="AMASS"``.
    """
    clip = load_motion_clip(path, dt=AMASS_DT)
    clip.source = "AMASS"
    return clip


def load_amass_dataset(
    directory: str | Path,
    subjects: list[str] | None = None,
) -> list[MotionClip]:
    """Load all AMASS-retargeted clips from *directory*.

    Parameters
    ----------
    directory:
        Directory containing ``.npz`` archives.
    subjects:
        If provided, keep only clips whose :attr:`~MotionClip.name` starts
        with one of the specified subject prefixes.

    Returns
    -------
    list[MotionClip]
        Clips tagged with ``source="AMASS"``.
    """
    clips = load_motion_clips(directory, dt=AMASS_DT)
    for clip in clips:
        clip.source = "AMASS"
    if subjects is not None:
        clips = [c for c in clips if any(c.name.startswith(s) for s in subjects)]
    return clips
