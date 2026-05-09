"""Pinocchio rigid-body dynamics wrapper for the Unitree G1 humanoid.

All Pinocchio imports are deferred to function bodies so that this module
can be imported without Pinocchio installed.  If Pinocchio is absent,
callers receive a clear :exc:`ImportError` with installation instructions.

Functions
---------
load_g1_model
    Load the Unitree G1 URDF into a Pinocchio model and data object.
mass_matrix_at
    Compute the mass matrix M(q) via Pinocchio CRBA.

Notes
-----
Pinocchio must be installed via the ``pin`` package::

    pip install pin

or via conda::

    conda install -c conda-forge pinocchio
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_g1_model(urdf_path: str | None = None) -> tuple[Any, Any]:
    """Load Unitree G1 URDF into a Pinocchio model and data pair.

    Parameters
    ----------
    urdf_path : str or None, optional
        Absolute or relative path to the G1 URDF file.  If ``None``, the
        function attempts to locate the URDF from the Genesis installation
        directory automatically.

    Returns
    -------
    model : pin.Model
        Pinocchio kinematic/dynamic model.
    data : pin.Data
        Pre-allocated data structure for algorithm outputs.

    Raises
    ------
    ImportError
        If ``pinocchio`` (``pin``) is not installed.
    FileNotFoundError
        If ``urdf_path`` is ``None`` and Genesis is not installed, so the
        URDF location cannot be determined automatically.

    Notes
    -----
    The G1 model includes a floating base (6-DOF free-flyer joint) plus
    23 actuated joints, giving ``nq = 7 + 23 = 30`` (quaternion
    representation) and ``nv = 6 + 23 = 29``.
    """
    try:
        import pinocchio as pin  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pinocchio (pin) is required. Install with: pip install pin"
        ) from exc

    if urdf_path is None:
        import importlib
        import os

        try:
            import genesis as gs  # type: ignore[import-not-found]  # noqa: F401

            genesis_file = importlib.import_module("genesis").__file__
            if genesis_file is None:
                raise ImportError("genesis module has no __file__")
            genesis_dir = os.path.dirname(genesis_file)
            urdf_path = os.path.join(genesis_dir, "assets", "urdf", "g1", "g1.urdf")
            logger.debug("Auto-detected G1 URDF path: %s", urdf_path)
        except ImportError as exc:
            raise FileNotFoundError(
                "Genesis not installed and no urdf_path provided. "
                "Install genesis-world or pass urdf_path explicitly."
            ) from exc

    logger.info("Loading G1 Pinocchio model from: %s", urdf_path)
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    logger.info(
        "Loaded G1 model: nq=%d, nv=%d, nbodies=%d",
        model.nq,
        model.nv,
        model.nbodies,
    )
    return model, data


def mass_matrix_at(
    model: Any,
    data: Any,
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute mass matrix M(q) via Pinocchio Composite-Rigid-Body Algorithm.

    Parameters
    ----------
    model : pin.Model
        Pinocchio kinematic/dynamic model (e.g. from :func:`load_g1_model`).
    data : pin.Data
        Pre-allocated data structure associated with ``model``.
    q : np.ndarray, shape (nq,)
        Joint configuration vector.  For the G1 floating-base model this
        has length ``nq = 30`` (4-DOF quaternion for the base + 23 joints).

    Returns
    -------
    M : np.ndarray, shape (nv, nv)
        Mass matrix, symmetric positive definite.  Size is ``nv x nv``
        where ``nv = 29`` for the G1 floating-base model.

    Notes
    -----
    Pinocchio's ``crba`` fills only the *upper-triangular* part of
    ``data.M``.  This function symmetrises the result before returning.

    The CRBA is an O(n) algorithm in the number of bodies.

    Examples
    --------
    >>> import numpy as np
    >>> # Requires pinocchio to be installed
    >>> # model, data = load_g1_model("/path/to/g1.urdf")
    >>> # q = np.zeros(model.nq); q[6] = 1.0  # unit quaternion
    >>> # M = mass_matrix_at(model, data, q)
    >>> # assert M.shape == (model.nv, model.nv)
    """
    try:
        import pinocchio as pin  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pinocchio (pin) is required. Install with: pip install pin"
        ) from exc

    q_arr = np.asarray(q, dtype=np.float64)
    pin.crba(model, data, q_arr)
    M = np.array(data.M, dtype=np.float64)

    # Symmetrise: CRBA returns upper-triangular; fill lower triangle
    M = M + M.T - np.diag(np.diag(M))
    return M
