"""Hydra structured configuration dataclasses for hamiltonian-modal.

All configs are plain Python dataclasses decorated with ``@dataclass`` so
they can be used as Hydra config nodes.  Default values match the numbers
reported in the paper.

Examples
--------
>>> from hamiltonian_modal.config import ModalConfig, WorldModelTrainConfig
>>> cfg = ModalConfig()
>>> cfg.n_modes
20
>>> train_cfg = WorldModelTrainConfig()
>>> train_cfg.batch_size
256
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModalConfig:
    """Configuration for the modal decomposition layer.

    Parameters
    ----------
    n_modes : int
        Number of structural vibration modes to retain.  Must be ≤ the
        number of velocity DOFs of the robot (23 for Unitree G1).
    n_contact_modes : int
        Number of additional contact-specific mode shapes appended to the
        modal basis during ground-contact phases.

    Notes
    -----
    Increasing ``n_modes`` improves expressiveness at the cost of a larger
    latent dimension for the Hamiltonian network.  The paper uses 20 + 5.
    """

    n_modes: int = 20
    n_contact_modes: int = 5


@dataclass
class HamiltonianNetConfig:
    """Architecture configuration for the Hamiltonian neural network.

    Parameters
    ----------
    v_hidden : List[int]
        Hidden layer widths for the potential-energy MLP V(q).
        The kinetic energy T = 0.5 * p^T M^{-1} p is computed analytically,
        so only V needs a network.
    contact_hidden : List[int]
        Hidden layer widths for the contact-event classifier head that
        predicts discrete contact transitions alongside the continuous
        Hamiltonian dynamics.

    Notes
    -----
    Both networks use SiLU activations and layer normalisation between
    hidden layers, following the architecture in the paper.
    """

    v_hidden: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    contact_hidden: List[int] = field(default_factory=lambda: [128, 128, 128])


@dataclass
class WorldModelTrainConfig:
    """Training hyper-parameters for the Hamiltonian-modal world model.

    Parameters
    ----------
    n_iterations : int
        Total number of gradient steps (paper: 100 000).
    batch_size : int
        Number of trajectory segments per mini-batch.
    learning_rate : float
        Initial Adam learning rate (cosine-annealed to 1e-5).
    grad_clip : float
        Global gradient-norm clipping threshold.
    sequence_length : int
        Length of trajectory segments in time-steps used for
        multi-step prediction loss.
    quantile_loss_weight : float
        Weight λ_q on the quantile regression loss for uncertainty
        estimation of the Hamiltonian prediction.
    contact_event_loss_weight : float
        Weight λ_c on the binary cross-entropy loss for contact-event
        classification.
    energy_conservation_loss_weight : float
        Weight λ_e on the soft energy-conservation penalty
        ||H(t+dt) - H(t)||².
    seed : int
        Global random seed for reproducibility.

    Notes
    -----
    The total loss is::

        L = L_pred + λ_q * L_quantile + λ_c * L_contact + λ_e * L_energy
    """

    n_iterations: int = 100_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    sequence_length: int = 50
    quantile_loss_weight: float = 0.1
    contact_event_loss_weight: float = 0.5
    energy_conservation_loss_weight: float = 0.01
    seed: int = 42


@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search recovery planning.

    Parameters
    ----------
    branching : int
        Number of child actions sampled at each node expansion.
    depth : int
        Maximum tree depth (i.e. planning horizon in time-steps).
    n_simulations : int
        Number of MCTS simulations (iterations of
        select → expand → rollout → backup) per planning call.
    c_puct : float
        Exploration constant in the PUCT formula::

            U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

    rollout_temperature : float
        Softmax temperature applied to the world-model action distribution
        during fast rollouts.  Set to 1.0 for on-policy rollouts;
        lower values make rollouts more greedy.

    Notes
    -----
    The world model is used as a simulator inside MCTS, so ``depth * dt``
    gives the effective planning horizon in seconds.
    """

    branching: int = 8
    depth: int = 50
    n_simulations: int = 50
    c_puct: float = 1.5
    rollout_temperature: float = 1.0


@dataclass
class DiffSimConfig:
    """Configuration for differentiable-simulation policy optimisation.

    Parameters
    ----------
    horizon : int
        Final trajectory optimisation horizon (time-steps).  The
        curriculum ramps from ``curriculum_start`` to this value.
    use_world_model : bool
        If ``True``, substitute the Genesis differentiable simulator
        with the learned Hamiltonian-modal world model during gradient
        computation (faster but approximate).
    curriculum_start : int
        Initial horizon length at the start of curriculum training.
        Must be ≤ ``curriculum_end``.
    curriculum_end : int
        Horizon at which curriculum training ends and the full
        ``horizon`` is used.  Typically equals ``horizon``.

    Notes
    -----
    The curriculum linearly interpolates the horizon over training
    iterations, following the schedule described in Section 4.2 of
    the paper.  A short initial horizon avoids exploding gradients
    that arise from long Genesis rollouts (cf. Fig. 3).
    """

    horizon: int = 100
    use_world_model: bool = False
    curriculum_start: int = 10
    curriculum_end: int = 100
