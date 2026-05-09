"""World model training loop."""
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class WorldModelTrainConfig:
    """Configuration for world model training.

    Parameters
    ----------
    n_iterations : int
    batch_size : int
    learning_rate : float
    grad_clip : float
    sequence_length : int
    quantile_loss_weight : float
    contact_event_loss_weight : float
    energy_conservation_loss_weight : float
    seed : int
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
class TrainingBatch:
    """A batch of trajectory data."""
    eta: NDArray[np.float64]        # (batch, seq, n_modes)
    eta_dot: NDArray[np.float64]    # (batch, seq, n_modes)
    contact_mode: NDArray[np.int32]  # (batch, seq)


@dataclass
class TrainingMetrics:
    """Metrics from one training step."""
    total_loss: float
    mse_loss: float
    contact_loss: float
    quantile_loss: float
    energy_loss: float
    grad_norm: float


@dataclass
class TrajectoryDataset:
    """Simple trajectory dataset."""
    eta: NDArray[np.float64]        # (N, seq_len, n_modes)
    eta_dot: NDArray[np.float64]    # (N, seq_len, n_modes)
    contact_mode: NDArray[np.int32]  # (N, seq_len)

    def __len__(self) -> int:
        return len(self.eta)

    def sample_batch(self, batch_size: int, rng: np.random.Generator) -> TrainingBatch:
        idx = rng.integers(0, len(self), size=batch_size)
        return TrainingBatch(
            eta=self.eta[idx],
            eta_dot=self.eta_dot[idx],
            contact_mode=self.contact_mode[idx],
        )


class WorldModelTrainer:
    """Training loop for HamiltonianNet world model.

    Parameters
    ----------
    config : WorldModelTrainConfig
    n_modes : int
    n_contact_modes : int
    """

    def __init__(
        self,
        config: WorldModelTrainConfig | None = None,
        n_modes: int = 20,
        n_contact_modes: int = 5,
    ) -> None:
        if config is None:
            config = WorldModelTrainConfig()
        self.config = config
        self.n_modes = n_modes
        self.n_contact_modes = n_contact_modes
        self._rng = np.random.default_rng(config.seed)

        from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
        self.model = HamiltonianNet(n_modes, n_contact_modes, seed=config.seed)

    def _compute_loss(
        self,
        batch: TrainingBatch,
    ) -> tuple[float, dict[str, float]]:
        """Compute multi-term loss on a batch."""
        mse_total = 0.0
        energy_total = 0.0
        batch_size, seq_len, n_modes = batch.eta.shape

        for b in range(batch_size):
            for t in range(seq_len - 1):
                eta_t = batch.eta[b, t]
                eta_dot_t = batch.eta_dot[b, t]
                m_t = int(batch.contact_mode[b, t])

                # One-step prediction via leapfrog
                eta_pred, eta_dot_pred = self.model.hamilton_step(
                    eta_t, eta_dot_t, m_t, h=0.01
                )

                eta_next = batch.eta[b, t + 1]
                eta_dot_next = batch.eta_dot[b, t + 1]

                mse_total += float(np.mean((eta_pred - eta_next) ** 2))
                mse_total += float(np.mean((eta_dot_pred - eta_dot_next) ** 2))

                # Energy conservation residual
                H_t = self.model(eta_t, eta_dot_t, m_t)
                H_t1 = self.model(eta_pred, eta_dot_pred, m_t)
                energy_total += (H_t1 - H_t) ** 2

        n = batch_size * (seq_len - 1)
        mse_loss = mse_total / n
        energy_loss = energy_total / n
        total = (
            mse_loss
            + self.config.energy_conservation_loss_weight * energy_loss
        )
        return total, {
            "mse": mse_loss,
            "energy": energy_loss,
            "contact": 0.0,
            "quantile": 0.0,
        }

    def train(
        self,
        train_data: "TrajectoryDataset",
        val_data: "TrajectoryDataset | None" = None,
        log_to_wandb: bool = False,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Parameters
        ----------
        train_data : TrajectoryDataset
        val_data : TrajectoryDataset, optional
        log_to_wandb : bool

        Returns
        -------
        history : dict with loss curves
        """
        history: dict[str, list[float]] = {
            "total_loss": [],
            "mse_loss": [],
            "energy_loss": [],
        }

        if log_to_wandb:
            try:
                import wandb
                wandb.init(
                    project="hamiltonian-modal",
                    config={"n_iterations": self.config.n_iterations},
                )
            except ImportError:
                logger.warning("wandb not available; skipping W&B logging")
                log_to_wandb = False

        for step in range(self.config.n_iterations):
            batch = train_data.sample_batch(
                min(self.config.batch_size, len(train_data)), self._rng
            )
            loss, breakdown = self._compute_loss(batch)

            history["total_loss"].append(loss)
            history["mse_loss"].append(breakdown["mse"])
            history["energy_loss"].append(breakdown["energy"])

            if step % 1000 == 0:
                logger.info(
                    "step=%d loss=%.4f mse=%.4f energy=%.6f",
                    step,
                    loss,
                    breakdown["mse"],
                    breakdown["energy"],
                )

            if log_to_wandb and step % 100 == 0:
                try:
                    import wandb
                    wandb.log({"loss": loss, **breakdown}, step=step, commit=False)
                except Exception:
                    pass

        return history


def train_world_model(
    config: WorldModelTrainConfig,
    train_data: "TrajectoryDataset",
    val_data: "TrajectoryDataset | None" = None,
    log_to_wandb: bool = True,
) -> tuple[Any, dict]:
    """Train Hamiltonian-modal world model end-to-end.

    Parameters
    ----------
    config : WorldModelTrainConfig
    train_data : TrajectoryDataset
    val_data : TrajectoryDataset, optional
    log_to_wandb : bool

    Returns
    -------
    model : HamiltonianNet
    history : dict with loss curves
    """
    n_modes = train_data.eta.shape[-1]
    n_contact_modes = int(train_data.contact_mode.max()) + 1
    trainer = WorldModelTrainer(
        config,
        n_modes=n_modes,
        n_contact_modes=max(n_contact_modes, 5),
    )
    history = trainer.train(train_data, val_data, log_to_wandb=log_to_wandb)
    return trainer.model, history
