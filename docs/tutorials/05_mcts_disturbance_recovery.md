# MCTS disturbance recovery

This tutorial demonstrates using Monte-Carlo Tree Search to plan push-recovery
actions for the Unitree G1.

## Setup

```python
import numpy as np
from hamiltonian_modal.config import MCTSConfig
from hamiltonian_modal.envs.g1_push_recovery import G1PushRecoveryEnv
from hamiltonian_modal.mcts.search import mcts_search
from hamiltonian_modal.policy.mlp_policy import MLPPolicy
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
from hamiltonian_modal.world_model.verifier import Verifier

env = G1PushRecoveryEnv()
n_dof = env.n_dof

# World model and verifier (both randomly initialised here — normally pre-trained)
params   = HamiltonianParams.random_init(n_modes=8, seed=0)
net      = HamiltonianNet(params)
verifier = Verifier(obs_dim=env.obs_dim, hidden_dim=64)
policy   = MLPPolicy(obs_dim=env.obs_dim, act_dim=n_dof, hidden_dim=64, n_layers=2)

mcts_cfg = MCTSConfig(n_simulations=20, c_puct=1.0, max_depth=10, discount=0.99)
```

## Run an episode with MCTS planning

```python
obs, _ = env.reset(seed=42)

total_reward = 0.0
for step in range(200):
    # MCTS selects the best action from the current observation
    action_idx = mcts_search(
        root_obs=obs,
        policy_fn=policy,
        value_fn=lambda o: verifier.value(o),
        env_step_fn=lambda o, a: env.step(a),
        config=mcts_cfg,
    )
    # Convert discrete index back to continuous action
    action = policy(obs)
    result = env.step(action)
    total_reward += result.reward
    obs = result.obs
    if result.terminated or result.truncated:
        break

print(f"Episode return: {total_reward:.2f}")
```

## Disturbance-recovery analysis

The `G1PushRecoveryEnv` applies random impulses during the episode.  You can
replay and analyse recovery quality:

```python
from hamiltonian_modal.world_model.contact_event import ContactEventClassifier

classifier = ContactEventClassifier(obs_dim=env.obs_dim, n_contacts=4, hidden_dim=32)
contact_probs = classifier(obs)
print("Contact probabilities:", contact_probs)
```

See the [bounded drift theorem](../theory/bounded_drift_theorem.md) for a
formal analysis of why symplectic world models remain stable under perturbations.
