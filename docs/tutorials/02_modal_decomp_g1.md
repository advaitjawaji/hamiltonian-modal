# Modal decomposition for G1

This tutorial shows how to decompose the Unitree G1's joint-space dynamics into
a low-dimensional modal subspace and use the encoding/decoding utilities.

## Background

The G1 has 23 revolute joints, giving a 23-DOF configuration space.  Working
directly in this space is expensive.  Modal decomposition projects dynamics
into *k ≪ 23* dominant vibrational modes while retaining most of the
energetically significant motion.

## Step 1 — Build the mass matrix

Use Pinocchio's CRBA algorithm (if installed) or a diagonal approximation:

```python
import numpy as np
from hamiltonian_modal.utils.pinocchio_wrapper import load_g1_model, mass_matrix_at

# With pinocchio + a G1 URDF:
# model, data = load_g1_model("/path/to/g1.urdf")
# M = mass_matrix_at(model, data, q=np.zeros(model.nq))

# Without pinocchio — diagonal approximation:
n_dof = 23
M = np.diag(np.ones(n_dof) * 2.0)   # 2 kg per joint (rough estimate)
```

## Step 2 — Build the stiffness matrix

```python
from hamiltonian_modal.modal import diagonal_stiffness, stiffness_from_gravity_hessian

# Option A: uniform spring stiffness
K = diagonal_stiffness(n_dof, k_values=5.0)

# Option B: linearise gravity around equilibrium
# def gravity_fn(q): return ...  # your gravity vector function
# q_eq = np.zeros(n_dof)
# K = stiffness_from_gravity_hessian(gravity_fn, q_eq, eps=1e-5)
```

## Step 3 — Compute the modal basis

```python
from hamiltonian_modal.modal import compute_modal_basis

basis = compute_modal_basis(M, K, n_modes=8)
print("Mode shapes:", basis.mode_shapes.shape)   # (23, 8)
print("Frequencies:", basis.frequencies)          # rad/s, ascending
```

## Step 4 — Encode and decode state

```python
from hamiltonian_modal.modal import (
    encode_position, encode_velocity, encode_momentum,
    decode_position, decode_velocity, decode_momentum,
)

q = np.random.randn(n_dof)   # physical joint positions
p = np.random.randn(n_dof)   # physical momenta

eta = encode_position(basis, M, q)   # modal coordinates (8,)
mu  = encode_momentum(basis, p)       # modal momenta    (8,)

q_reconstructed = decode_position(basis, eta)   # (23,)
```

## Step 5 — Switch modes per locomotion phase

```python
from hamiltonian_modal.modal import select_low_frequency_modes, select_modes_by_frequency_range

# Use only the 3 lowest-frequency modes for slow walking
walking_basis = select_low_frequency_modes(basis, n_modes=3)

# Band-pass: select modes active in running gait (2–15 rad/s)
running_basis = select_modes_by_frequency_range(basis, f_min=2.0, f_max=15.0)
```

See the [theory](../theory/modal_decomposition.md) for the mathematical details.
