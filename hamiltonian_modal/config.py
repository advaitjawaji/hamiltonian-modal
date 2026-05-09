"""Structured configuration entry points for Hamiltonian Modal experiments.

All configuration objects are plain :mod:`dataclasses` so they can be
serialised with :mod:`json` or replaced by `hydra`/`omegaconf` later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "ModalConfig",
    "HamiltonianNetConfig",
    "SymplecticConfig",
    "MCTSConfig",
    "MLPPolicyConfig",
    "ILQRConfig",
    "DDPConfig",
    "DiffSimConfig",
    "EnvConfig",
    "ExperimentConfig",
]


@dataclass
class ModalConfig:
    """Configuration for the modal decomposition layer."""

    n_modes: int = 10
    """Number of retained modal modes (k ≤ n_dof)."""
    fd_eps: float = 1e-5
    """Finite-difference step for stiffness-from-gravity-hessian."""


@dataclass
class HamiltonianNetConfig:
    """Configuration for the Hamiltonian neural network."""

    hidden_dim: int = 256
    """Width of each hidden layer."""
    n_layers: int = 3
    """Number of hidden layers."""
    activation: str = "tanh"
    """Activation name: ``"tanh"`` or ``"relu"``."""


@dataclass
class SymplecticConfig:
    """Configuration for symplectic integration."""

    integrator: str = "leapfrog"
    """Integrator variant: ``"leapfrog"`` or ``"stormer_verlet"``."""
    dt: float = 0.001
    """Integration timestep in seconds."""
    n_substeps: int = 1
    """Number of sub-steps per environment step."""


@dataclass
class MCTSConfig:
    """Configuration for Monte-Carlo Tree Search."""

    n_simulations: int = 50
    """Number of MCTS rollouts per decision."""
    c_puct: float = 1.0
    """Exploration constant for PUCT."""
    max_depth: int = 20
    """Maximum tree depth."""
    discount: float = 0.99
    """Discount factor for value backup."""


@dataclass
class MLPPolicyConfig:
    """Configuration for MLP-based policy."""

    hidden_dim: int = 256
    n_layers: int = 2
    activation: str = "tanh"


@dataclass
class ILQRConfig:
    """Configuration for iterative LQR."""

    n_iterations: int = 10
    """Maximum number of iLQR passes."""
    horizon: int = 50
    """Planning horizon (steps)."""
    reg_init: float = 1.0
    """Initial regularisation value."""
    reg_min: float = 1e-6
    reg_max: float = 1e10
    line_search_beta: float = 0.5
    """Backtracking line-search decay factor."""


@dataclass
class DDPConfig(ILQRConfig):
    """Configuration for Differential Dynamic Programming (extends iLQR)."""

    second_order_value: bool = True
    """Whether to include second-order value-function terms."""


@dataclass
class DiffSimConfig:
    """Configuration for the differentiable simulation pipeline."""

    n_epochs: int = 100
    horizon: int = 200
    batch_size: int = 8
    lr: float = 1e-3
    grad_clip: float = 1.0
    """Maximum gradient norm (0 = disabled)."""


@dataclass
class EnvConfig:
    """Configuration shared by all G1 environments."""

    dt: float = 0.005
    """Simulation timestep in seconds."""
    episode_length: int = 1000
    """Maximum steps per episode."""
    action_scale: float = 1.0
    """Scale applied to raw policy outputs before sending to robot."""
    obs_noise_std: float = 0.0
    """Standard deviation of Gaussian noise added to observations."""


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration aggregating all sub-configs."""

    modal: ModalConfig = field(default_factory=ModalConfig)
    hamiltonian: HamiltonianNetConfig = field(default_factory=HamiltonianNetConfig)
    symplectic: SymplecticConfig = field(default_factory=SymplecticConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    policy: MLPPolicyConfig = field(default_factory=MLPPolicyConfig)
    ilqr: ILQRConfig = field(default_factory=ILQRConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    diff_sim: DiffSimConfig = field(default_factory=DiffSimConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 0
    """Global random seed for reproducibility."""
