"""Genesis-modal-Hamiltonian pipeline interfaces.

Provides :class:`DiffSimPipeline` — the end-to-end rollout engine that:

1. Resets a G1 environment.
2. Encodes the physical state into modal coordinates.
3. Integrates the Hamiltonian world model with a symplectic integrator.
4. Decodes modal predictions back to joint space.
5. Returns the full trajectory for policy-gradient or iLQR use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hamiltonian_modal.config import DiffSimConfig, SymplecticConfig
from hamiltonian_modal.envs.g1_base import G1BaseEnv
from hamiltonian_modal.modal.decomposition import ModalBasis
from hamiltonian_modal.modal.decoder import decode_position, decode_velocity
from hamiltonian_modal.modal.encoder import encode_momentum, encode_position, encode_velocity
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
from hamiltonian_modal.world_model.symplectic import leapfrog_step

__all__ = [
    "Trajectory",
    "DiffSimPipeline",
]


@dataclass
class Trajectory:
    """Output of one :class:`DiffSimPipeline` rollout.

    Attributes
    ----------
    etas:
        Modal positions, shape ``(T+1, n_modes)``.
    mus:
        Modal momenta, shape ``(T+1, n_modes)``.
    qs:
        Physical joint positions, shape ``(T+1, n_dof)``.
    vs:
        Physical joint velocities, shape ``(T+1, n_dof)``.
    rewards:
        Per-step rewards, shape ``(T,)``.
    total_return:
        Sum of undiscounted rewards.
    """

    etas: NDArray[np.float64]
    mus: NDArray[np.float64]
    qs: NDArray[np.float64]
    vs: NDArray[np.float64]
    rewards: NDArray[np.float64]

    @property
    def total_return(self) -> float:
        return float(np.sum(self.rewards))


class DiffSimPipeline:
    """End-to-end differentiable simulation pipeline.

    Parameters
    ----------
    env:
        A :class:`~hamiltonian_modal.envs.g1_base.G1BaseEnv` instance.
    basis:
        Pre-computed :class:`~hamiltonian_modal.modal.decomposition.ModalBasis`.
    M:
        Mass matrix at the nominal configuration, shape ``(n_dof, n_dof)``.
    ham_net:
        :class:`~hamiltonian_modal.world_model.hamiltonian_net.HamiltonianNet`.
    sim_config:
        :class:`~hamiltonian_modal.config.DiffSimConfig`.
    symp_config:
        :class:`~hamiltonian_modal.config.SymplecticConfig`.
    """

    def __init__(
        self,
        env: G1BaseEnv,
        basis: ModalBasis,
        M: NDArray[np.float64],
        ham_net: HamiltonianNet,
        sim_config: DiffSimConfig | None = None,
        symp_config: SymplecticConfig | None = None,
    ) -> None:
        self.env = env
        self.basis = basis
        self.M = M
        self.ham_net = ham_net
        self.sim_config = sim_config or DiffSimConfig()
        self.symp_config = symp_config or SymplecticConfig()

    def rollout(
        self,
        policy_fn,
        horizon: int | None = None,
        seed: int | None = None,
    ) -> Trajectory:
        """Execute one episode using *policy_fn* and the Hamiltonian world model.

        Parameters
        ----------
        policy_fn:
            Callable ``obs → action`` (e.g. an :class:`~hamiltonian_modal.policy.mlp_policy.MLPPolicy`).
        horizon:
            Maximum number of steps.  Defaults to ``sim_config.horizon``.
        seed:
            Optional reset seed.

        Returns
        -------
        Trajectory
        """
        T = horizon or self.sim_config.horizon
        obs, _ = self.env.reset(seed=seed)

        # Initialise state from env
        assert self.env._state is not None
        q0 = self.env._state.q.copy()
        v0 = self.env._state.v.copy()
        p0 = self.M @ v0

        # Encode initial modal state
        eta = encode_position(self.basis, self.M, q0)
        mu = encode_momentum(self.basis, p0)

        etas = [eta.copy()]
        mus = [mu.copy()]
        qs = [q0.copy()]
        vs = [v0.copy()]
        rewards = []

        dt = self.symp_config.dt

        def grad_H_eta(e: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.ham_net.grad_H_eta(e, mu)

        def grad_H_mu(m: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.ham_net.grad_H_mu(eta, m)

        for _ in range(T):
            action = policy_fn(obs)
            # Symplectic step in modal space
            eta, mu = leapfrog_step(eta, mu, grad_H_eta, grad_H_mu, dt)
            # Decode back to physical space
            q_new = decode_position(self.basis, eta)
            v_new = decode_velocity(self.basis, mu)  # mu ≈ eta_dot for unit-mass modal

            # Update env state to decoded physical state (override dynamics)
            from hamiltonian_modal.envs.g1_base import EnvState

            self.env._state = EnvState(
                q=q_new, v=v_new, t=self.env._state.t + dt, step=self.env._state.step + 1
            )
            result = self.env.step(action)
            rewards.append(result.reward)
            obs = result.obs
            etas.append(eta.copy())
            mus.append(mu.copy())
            qs.append(q_new.copy())
            vs.append(v_new.copy())

            if result.terminated or result.truncated:
                break

        return Trajectory(
            etas=np.array(etas),
            mus=np.array(mus),
            qs=np.array(qs),
            vs=np.array(vs),
            rewards=np.array(rewards),
        )
