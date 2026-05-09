# World model training

This tutorial covers training the `HamiltonianNet` world model from rollout data.

## Overview

The world model predicts modal-space dynamics:

```
H(η, μ) = T(μ) + V(η)
```

where **T** is a learned kinetic-energy quadratic form and **V** is a small MLP
potential.  We train it by minimising the symplectic-integration prediction error
on (η, μ, H*) triples.

## Collect training data

```python
import numpy as np
from hamiltonian_modal.envs.g1_walking import G1WalkingEnv
from hamiltonian_modal.modal import compute_modal_basis, diagonal_stiffness
from hamiltonian_modal.modal.encoder import encode_position, encode_momentum

env = G1WalkingEnv()
n_dof = env.n_dof
M = np.eye(n_dof)
K = diagonal_stiffness(n_dof, k_values=5.0)
basis = compute_modal_basis(M, K, n_modes=8)

data = []
obs, _ = env.reset(seed=0)
for _ in range(500):
    action = env.action_space_sample()
    step_result = env.step(action)
    q = env._state.q
    p = M @ env._state.v
    eta = encode_position(basis, M, q)
    mu  = encode_momentum(basis, p)
    data.append((eta, mu))
    obs = step_result.obs
    if step_result.done:
        obs, _ = env.reset()
```

## Train the world model

```python
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
from hamiltonian_modal.world_model.trainer import WorldModelTrainer
from hamiltonian_modal.config import HamiltonianNetConfig

params = HamiltonianParams.random_init(n_modes=8, hidden_dim=64, n_layers=2, seed=0)
net    = HamiltonianNet(params)

trainer = WorldModelTrainer(net, lr=1e-3, grad_clip=1.0)

etas = np.stack([d[0] for d in data])
mus  = np.stack([d[1] for d in data])
H_targets = np.array([net(e, m) for e, m in zip(etas, mus)])  # pseudo-labels

for epoch in range(50):
    metrics = trainer.train_epoch(etas, mus, H_targets)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")
```

## Evaluate uncertainty

```python
from hamiltonian_modal.world_model.uncertainty import EnsembleUncertainty

ensemble = EnsembleUncertainty(
    [HamiltonianNet(HamiltonianParams.random_init(8, seed=i)) for i in range(5)]
)
eta_test = np.zeros(8)
mu_test  = np.ones(8)
mean_H, var_H = ensemble(eta_test, mu_test)
print(f"Predicted H = {mean_H:.3f} ± {var_H**0.5:.3f}")
```

See the [Hamiltonian dynamics theory](../theory/hamiltonian_dynamics.md) for
the mathematical foundations.
