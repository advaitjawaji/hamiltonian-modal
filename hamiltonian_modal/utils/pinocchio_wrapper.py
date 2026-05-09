"""Pinocchio helper utilities for Unitree G1 model loading and dynamics."""

from __future__ import annotations

from importlib import import_module, resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pinocchio as pin


def _load_pinocchio() -> Any:
    try:
        return import_module("pinocchio")
    except ImportError as exc:  # pragma: no cover - depends on local environment
        msg = (
            "pinocchio is required for this operation. Install the project with robotics "
            "dependencies to use `hamiltonian_modal.utils.pinocchio_wrapper`."
        )
        raise ImportError(msg) from exc


def _default_g1_urdf_path() -> Path:
    try:
        genesis_root = resources.files("genesis")
    except (ModuleNotFoundError, TypeError) as exc:  # pragma: no cover - env dependent
        raise FileNotFoundError(
            "Could not discover Genesis package assets. Pass `urdf_path` explicitly."
        ) from exc

    candidates = (
        "assets/robots/g1/g1.urdf",
        "assets/robots/unitree_g1/g1.urdf",
        "assets/urdf/g1.urdf",
    )
    for relative in candidates:
        candidate = genesis_root / relative
        if candidate.is_file():
            return Path(candidate)

    raise FileNotFoundError("No bundled G1 URDF found in Genesis assets.")


def load_g1_model(urdf_path: str | None = None) -> tuple["pin.Model", "pin.Data"]:
    """Load Unitree G1 URDF into Pinocchio.

    Parameters
    ----------
    urdf_path : str | None, optional
        Explicit URDF path. If None, uses a Genesis-bundled G1 URDF path.

    Returns
    -------
    tuple[pin.Model, pin.Data]
        Pinocchio model and associated data object.
    """

    pin = _load_pinocchio()
    resolved_path = Path(urdf_path) if urdf_path is not None else _default_g1_urdf_path()
    model = pin.buildModelFromUrdf(str(resolved_path))
    data = model.createData()
    return model, data


def mass_matrix_at(model: "pin.Model", data: "pin.Data", q: ArrayLike) -> NDArray[np.float64]:
    """Compute the generalized mass matrix M(q) via Pinocchio CRBA.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model.
    data : pin.Data
        Pinocchio data associated with the model.
    q : ArrayLike
        Joint configuration with shape ``(model.nq,)``.

    Returns
    -------
    NDArray[np.float64]
        Symmetric mass matrix of shape ``(model.nv, model.nv)``.
    """

    pin = _load_pinocchio()
    q_array = np.asarray(q, dtype=np.float64).reshape(-1)
    if q_array.shape[0] != model.nq:
        raise ValueError(f"Expected q with size {model.nq}, got {q_array.shape[0]}.")

    mass_matrix = np.asarray(pin.crba(model, data, q_array), dtype=np.float64)
    return 0.5 * (mass_matrix + mass_matrix.T)

