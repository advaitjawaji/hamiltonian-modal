# Quickstart

## Installation

```bash
pip install hamiltonian-modal
```

For JAX support (GPU recommended):

```bash
pip install "hamiltonian-modal[jax]"
```

## Train a World Model

```python
import numpy as np
import hamiltonian_modal as hm
from hamiltonian_modal.world_model.trainer import WorldModelTrainConfig, TrajectoryDataset, train_world_model

# Build synthetic dataset (replace with real G1 trajectories)
rng = np.random.default_rng(42)
train_data = TrajectoryDataset(
    eta=rng.standard_normal((1000, 50, 20)),
    eta_dot=rng.standard_normal((1000, 50, 20)),
    contact_mode=rng.integers(0, 5, size=(1000, 50)).astype(np.int32),
)

config = WorldModelTrainConfig(n_iterations=10_000, n_modes=20)
model, history = train_world_model(config, train_data)
print(f"Final loss: {history['total_loss'][-1]:.4f}")
```

## Run MCTS Recovery

```python
from hamiltonian_modal.mcts.search import MCTS, MCTSConfig, ModalState, Goal
from hamiltonian_modal.world_model.verifier import Verifier

verifier = Verifier(n_modes=20, n_contact_modes=5)

def policy(obs):
    mean = np.zeros(23)
    cov = np.eye(23) * 0.01
    return mean, cov

mcts = MCTS(model, verifier, policy, MCTSConfig(depth=50, n_simulations=50))
root_state = ModalState(eta=np.zeros(20), eta_dot=np.zeros(20))
goal = Goal(target_position=np.array([10.0, 0.0, 0.0]))
plan = mcts.search(root_state, goal)
print(f"MCTS expected value: {plan.expected_value:.4f}, depth: {plan.search_depth_reached}")
```

## CLI Scripts

```bash
# Train world model
python scripts/train_world_model.py --n-iterations 100000

# Train diff-sim policy
python scripts/train_diff_sim_policy.py --n-iterations 10000

# Run MCTS recovery evaluation
python scripts/run_mcts_recovery.py --n-trials 100

# Run DriftBench
python scripts/run_driftbench.py

# Run EffBench
python scripts/run_diffsim_effbench.py

# Reproduce all paper figures
python scripts/reproduce_paper_figures.py
```
