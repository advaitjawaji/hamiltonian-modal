from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hamiltonian_modal.utils.pinocchio_wrapper import load_g1_model, mass_matrix_at


@dataclass
class _FakeData:
    pass


@dataclass
class _FakeModel:
    nq: int
    nv: int

    def createData(self) -> _FakeData:
        return _FakeData()


def _install_fake_pinocchio(monkeypatch: pytest.MonkeyPatch) -> None:
    def _build_model_from_urdf(_path: str) -> _FakeModel:
        return _FakeModel(nq=3, nv=3)

    def _crba(_model: _FakeModel, _data: _FakeData, _q: np.ndarray) -> np.ndarray:
        return np.array([[3.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 1.5]])

    monkeypatch.setitem(
        sys.modules,
        "pinocchio",
        SimpleNamespace(buildModelFromUrdf=_build_model_from_urdf, crba=_crba),
    )


def test_load_g1_model_with_explicit_urdf_uses_pinocchio(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pinocchio(monkeypatch)
    model, data = load_g1_model(str(Path("/tmp/g1.urdf")))
    assert isinstance(model, _FakeModel)
    assert isinstance(data, _FakeData)


def test_mass_matrix_is_symmetric_positive_definite(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pinocchio(monkeypatch)
    model = _FakeModel(nq=3, nv=3)
    data = _FakeData()
    matrix = mass_matrix_at(model, data, q=np.array([0.1, -0.2, 0.3]))
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(matrix) > 0.0)


def test_mass_matrix_rejects_wrong_q_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pinocchio(monkeypatch)
    model = _FakeModel(nq=3, nv=3)
    data = _FakeData()
    with pytest.raises(ValueError, match="Expected q with size 3"):
        mass_matrix_at(model, data, q=np.array([0.1, 0.2]))
