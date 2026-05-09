# Tutorial 4: End-to-End Diff-Sim Policy Gradient

## Overview

Gradient descent is computed through the entire stack:
policy → world model → modal projection → Genesis dynamics → policy params.

The modal-Hamiltonian layer is what makes long-horizon backprop stable.

## Building the Pipeline

```python
import numpy as np
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
from hamiltonian_modal.policy.mlp_policy import MLPPolicy
from hamiltonian_modal.diff_sim.pipeline import DiffSimPipeline

n_modes = 20
world_model = HamiltonianNet(n_modes, n_contact_modes=5, seed=42)
policy = MLPPolicy(obs_dim=n_modes * 2, action_dim=23, seed=42)

pipeline = DiffSimPipeline(
    policy=policy,
    world_model=world_model,
    n_modes=n_modes,
    n_contact_modes=5,
)
```

## Rollout

```python
rng = np.random.default_rng(42)
initial_obs = rng.standard_normal(n_modes * 2)

def reward_fn(state, action):
    return -float(np.mean(state[:n_modes] ** 2))

result = pipeline.rollout(initial_obs, horizon=50, use_world_model=True, reward_fn=reward_fn)
print(f"Total reward: {result.total_reward:.4f}")
```

## Trajectory Length Curriculum

```python
from hamiltonian_modal.diff_sim.stability_utils import trajectory_length_curriculum

for step in range(10_000):
    horizon = trajectory_length_curriculum(step, 10_000, t_start=10, t_end=100)
    # ... rollout with this horizon
```
