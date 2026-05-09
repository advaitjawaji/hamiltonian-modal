# Tutorial 2: Modal Decomposition for G1

## Compute Mode Shapes

```python
import numpy as np
from hamiltonian_modal.modal.decomposition import compute_modal_decomposition
from hamiltonian_modal.modal.stiffness import assemble_stiffness_matrix

# Synthetic mass matrix (replace with Pinocchio CRBA for real G1)
n = 23
rng = np.random.default_rng(42)
A = rng.standard_normal((n, n))
M = A @ A.T + 5 * np.eye(n)

# PD-gain dominated stiffness
K = assemble_stiffness_matrix(model=None, q_ref=np.zeros(n))

eigenvalues, mode_shapes = compute_modal_decomposition(M, K, n_modes=20)
omega = np.sqrt(eigenvalues) / (2 * np.pi)
print("Natural frequencies (Hz):", omega)
```

## Encode / Decode

```python
from hamiltonian_modal.modal.encoder import project_to_modal
from hamiltonian_modal.modal.decoder import project_from_modal

q = rng.standard_normal(n)
qdot = rng.standard_normal(n)
q_ref = np.zeros(n)

eta, eta_dot = project_to_modal(q, qdot, mode_shapes, M, q_ref)
q_rec, qdot_rec = project_from_modal(eta, eta_dot, mode_shapes, q_ref)

# With full basis, round-trip is exact
assert np.allclose(q, q_rec, atol=1e-8)
```

## Verify Mass Orthonormality

```python
from hamiltonian_modal.modal.decomposition import is_m_orthonormal

assert is_m_orthonormal(mode_shapes, M)
```
