# Tutorial 3: World Model Training

## Overview

The world model $H_\theta(\eta, \dot{\eta}, m) = T + V_\theta + \Delta_\theta$ is trained
to predict modal-state transitions while maintaining energy conservation.

## Training Loop

```python
from hamiltonian_modal.world_model.trainer import (
    WorldModelTrainConfig,
    TrajectoryDataset,
    train_world_model,
)
import numpy as np

rng = np.random.default_rng(42)
n_modes = 20

train_data = TrajectoryDataset(
    eta=rng.standard_normal((500, 50, n_modes)),
    eta_dot=rng.standard_normal((500, 50, n_modes)),
    contact_mode=rng.integers(0, 5, size=(500, 50)).astype(np.int32),
)

config = WorldModelTrainConfig(
    n_iterations=50_000,
    batch_size=128,
    learning_rate=3e-4,
    energy_conservation_loss_weight=0.01,
    seed=42,
)

model, history = train_world_model(config, train_data, log_to_wandb=False)
```

## Rollout with Leapfrog

```python
import numpy as np

eta0 = np.zeros(n_modes)
eta_dot0 = rng.standard_normal(n_modes) * 0.1

eta, eta_dot = eta0.copy(), eta_dot0.copy()
H0 = model(eta, eta_dot, m=0)

for _ in range(50):
    eta, eta_dot = model.hamilton_step(eta, eta_dot, m=0, h=0.01)

H50 = model(eta, eta_dot, m=0)
drift = abs(H50 - H0) / abs(H0)
print(f"Energy drift at depth 50: {drift:.4%}")
```

## Uncertainty Estimation

```python
from hamiltonian_modal.world_model.uncertainty import QuantileUncertainty

uq = QuantileUncertainty(n_modes=n_modes, n_contact_modes=5)
estimate = uq(eta=eta0, m=0)
lo, hi = estimate.interval_90
print(f"90% prediction interval width: {(hi - lo).mean():.4f}")
```
