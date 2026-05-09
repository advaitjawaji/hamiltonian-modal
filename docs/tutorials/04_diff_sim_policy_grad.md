# Differentiable simulation policy gradients

This tutorial shows how to train a walking policy for the G1 using REINFORCE
through the `DiffSimPipeline`.

## Setup

```python
import numpy as np
from hamiltonian_modal.config import DiffSimConfig, SymplecticConfig
from hamiltonian_modal.envs.g1_walking import G1WalkingEnv
from hamiltonian_modal.modal import compute_modal_basis, diagonal_stiffness
from hamiltonian_modal.policy.mlp_policy import MLPPolicy
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
from hamiltonian_modal.diff_sim.pipeline import DiffSimPipeline
from hamiltonian_modal.diff_sim.trainer import DiffSimTrainer

n_dof = 23
M = np.eye(n_dof)
K = diagonal_stiffness(n_dof, k_values=5.0)
basis = compute_modal_basis(M, K, n_modes=8)

env    = G1WalkingEnv()
params = HamiltonianParams.random_init(n_modes=8, seed=0)
net    = HamiltonianNet(params)
policy = MLPPolicy(obs_dim=env.obs_dim, act_dim=n_dof, hidden_dim=64, n_layers=2)

pipeline = DiffSimPipeline(
    env=env,
    basis=basis,
    M=M,
    ham_net=net,
    sim_config=DiffSimConfig(horizon=50, batch_size=4),
    symp_config=SymplecticConfig(dt=0.01),
)
```

## Train with REINFORCE

```python
from hamiltonian_modal.diff_sim.policy_gradient import REINFORCEConfig, compute_returns, reinforce_update

pg_config = REINFORCEConfig(discount=0.99, use_baseline=True, lr=1e-3, grad_clip=1.0)

for epoch in range(100):
    trajectory = pipeline.rollout(policy_fn=policy, horizon=50, seed=epoch)
    returns = compute_returns(trajectory.rewards, discount=pg_config.discount)
    # Uniform log-probs (placeholder — replace with actual policy log-probs)
    log_probs = np.full_like(trajectory.rewards, -0.5 * n_dof * np.log(2 * np.pi))
    loss, grad_norm = reinforce_update(log_probs, returns, pg_config)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}  return={trajectory.total_return:.2f}  loss={loss:.4f}  grad={grad_norm:.4f}")
```

## Monitor gradient health

```python
from hamiltonian_modal.diff_sim.stability_utils import RunningMeanStd, gradient_checksum

rms = RunningMeanStd(shape=(n_dof,))
rms.update(trajectory.qs[1:])           # update observation statistics
obs_normed = rms.normalize(trajectory.qs[-1])
print("Normalised final state:", obs_normed[:4])
```

See the [symplectic integration theory](../theory/symplectic_integration.md) for
why the leapfrog integrator keeps gradients stable over long horizons.
