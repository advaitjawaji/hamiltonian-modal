# Tutorial 5: MCTS Disturbance Recovery

## Overview

MCTS with PUCT selection operates in the Hamiltonian-modal world model.
Depth-50 search is feasible because energy drift is bounded (<5%).

## Setup

```python
import numpy as np
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet
from hamiltonian_modal.world_model.verifier import Verifier
from hamiltonian_modal.mcts.search import MCTS, MCTSConfig, ModalState, Goal

n_modes = 20
world_model = HamiltonianNet(n_modes, n_contact_modes=5, seed=42)
verifier = Verifier(n_modes, n_contact_modes=5, goal_dim=3, seed=42)

def policy(obs):
    mean = np.zeros(23) * 0.1
    cov = np.eye(23) * 0.01
    return mean, cov

config = MCTSConfig(branching=8, depth=50, n_simulations=50, c_puct=1.5)
mcts = MCTS(world_model, verifier, policy, config)
```

## Search

```python
root_state = ModalState(
    eta=np.zeros(n_modes),
    eta_dot=np.zeros(n_modes),
    contact_mode=0,
)
goal = Goal(target_position=np.array([10.0, 0.0, 0.0]), success_radius=0.5)

plan = mcts.search(root_state, goal)
print(f"Expected value: {plan.expected_value:.4f}")
print(f"Search depth reached: {plan.search_depth_reached}")
print(f"Actions planned: {len(plan.actions)}")
```

## Disturbance Recovery Environment

```python
from hamiltonian_modal.envs.g1_disturbance import G1DisturbanceEnv, DisturbanceConfig

env = G1DisturbanceEnv(DisturbanceConfig(disturbance_time=4.0))
obs = env.reset()

for step in range(200):
    eta = obs[:n_modes]
    eta_dot = obs[n_modes:n_modes*2] if len(obs) >= n_modes*2 else np.zeros(n_modes)
    root = ModalState(eta=eta, eta_dot=eta_dot)
    plan = mcts.search(root, goal)
    action = plan.actions[0] if plan.actions else np.zeros(23)
    obs, reward, done, info = env.step(action)
    if env.reached_target():
        print(f"Target reached at step {step}!")
        break
    if done:
        break
```
