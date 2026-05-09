# Quickstart

This guide walks you through installing `hamiltonian-modal` and running your first
Hamiltonian-modal simulation on the Unitree G1 humanoid robot.

## Installation

```bash
# Install in editable mode with dev dependencies
python -m pip install -e '.[dev]'
```

## Verify the installation

```python
import hamiltonian_modal
print(hamiltonian_modal.__version__)  # 0.1.0
```

## Compute a modal basis

```python
import numpy as np
from hamiltonian_modal.modal import compute_modal_basis, diagonal_stiffness

n_dof = 23   # Unitree G1 joint DOF
M = np.eye(n_dof)  # identity mass matrix (replace with CRBA output)
K = diagonal_stiffness(n_dof, k_values=10.0)

basis = compute_modal_basis(M, K, n_modes=10)
print(f"Retained {basis.n_modes} modes from {basis.n_dof} DOF system")
print(f"Lowest frequency: {basis.frequencies[0]:.3f} rad/s")
```

## Run a Hamiltonian world-model rollout

```python
from hamiltonian_modal.world_model.hamiltonian_net import HamiltonianNet, HamiltonianParams
from hamiltonian_modal.world_model.symplectic import leapfrog_step

params = HamiltonianParams.random_init(n_modes=10, seed=0)
net = HamiltonianNet(params)

eta = np.zeros(10)   # modal position
mu  = np.ones(10)    # modal momentum

for _ in range(100):
    eta, mu = leapfrog_step(
        eta, mu,
        grad_H_q=lambda e: net.grad_H_eta(e, mu),
        grad_H_p=lambda m: net.grad_H_mu(eta, m),
        dt=0.01,
    )
print("Final Hamiltonian:", net(eta, mu))
```

## Next steps

* [Modal decomposition for G1](02_modal_decomp_g1.md)
* [World model training](03_world_model_training.md)
* [Differentiable simulation policy gradients](04_diff_sim_policy_grad.md)
* [MCTS disturbance recovery](05_mcts_disturbance_recovery.md)
