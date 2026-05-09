"""Uncertainty-head interfaces for the world model.

Implements an *ensemble* uncertainty estimator that wraps several
:class:`~hamiltonian_modal.world_model.hamiltonian_net.HamiltonianNet`
instances and exposes mean / variance of their predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams

__all__ = [
    "EnsembleUncertainty",
    "UncertaintyEstimate",
]


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a single (η, μ) query.

    Attributes
    ----------
    mean_H:
        Mean Hamiltonian value across ensemble members.
    var_H:
        Variance of Hamiltonian values across ensemble members.
    std_H:
        Standard deviation (√var_H).
    member_values:
        Individual Hamiltonian values from each ensemble member.
    """

    mean_H: float
    var_H: float
    std_H: float
    member_values: NDArray[np.float64]


class EnsembleUncertainty:
    """Ensemble of Hamiltonian networks for uncertainty quantification.

    Parameters
    ----------
    members:
        List of :class:`~hamiltonian_modal.world_model.hamiltonian_net.HamiltonianNet`
        instances.
    """

    def __init__(self, members: list[HamiltonianNet]) -> None:
        if not members:
            raise ValueError("Ensemble must contain at least one member.")
        self.members = members

    @classmethod
    def random_init(
        cls,
        n_members: int,
        n_modes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation: str = "tanh",
        seed: int = 0,
    ) -> "EnsembleUncertainty":
        """Initialise a random ensemble.

        Parameters
        ----------
        n_members:
            Number of ensemble members.
        n_modes:
            Number of modal coordinates.
        hidden_dim:
            Hidden-layer width for the potential MLP.
        n_layers:
            Number of hidden layers.
        activation:
            Activation function name.
        seed:
            Base seed; each member gets seed + i.

        Returns
        -------
        EnsembleUncertainty
        """
        members = [
            HamiltonianNet(
                HamiltonianParams.random_init(n_modes, hidden_dim, n_layers, seed=seed + i),
                activation=activation,
            )
            for i in range(n_members)
        ]
        return cls(members)

    @property
    def n_members(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    def predict(
        self,
        eta: NDArray[np.float64],
        mu: NDArray[np.float64],
    ) -> UncertaintyEstimate:
        """Query all ensemble members and return mean/variance.

        Parameters
        ----------
        eta:
            Modal position, shape ``(n_modes,)``.
        mu:
            Modal momentum, shape ``(n_modes,)``.

        Returns
        -------
        UncertaintyEstimate
        """
        values = np.array([m(eta, mu) for m in self.members], dtype=np.float64)
        mean_H = float(np.mean(values))
        var_H = float(np.var(values, ddof=1)) if self.n_members > 1 else 0.0
        return UncertaintyEstimate(
            mean_H=mean_H,
            var_H=var_H,
            std_H=float(np.sqrt(var_H)),
            member_values=values,
        )
