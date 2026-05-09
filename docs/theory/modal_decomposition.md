# Modal decomposition

## Overview

The **modal decomposition** layer converts the full *n*-dimensional joint
space of the Unitree G1 into a low-dimensional *modal* subspace that captures
the dominant structural dynamics.  This is the central abstraction of Phase 2.

---

## Generalised eigenvalue problem

Given a symmetric positive-definite mass matrix **M**(q) ∈ ℝⁿˣⁿ (computed by
Pinocchio's CRBA; see Phase 1) and a symmetric stiffness matrix **K** ∈ ℝⁿˣⁿ,
the *k* retained mode shapes **Φ** ∈ ℝⁿˣᵏ are the solution to:

```
K Φ = M Φ Λ
```

where **Λ** = diag(λ₁, …, λₖ) contains the *k* smallest eigenvalues.
The mode shapes are **M-orthonormal**:

```
Φᵀ M Φ = I_k
```

Natural frequencies are ω_i = √λ_i [rad s⁻¹].

### Numerical solution (Cholesky reduction)

No external dependency beyond NumPy is required:

1. Cholesky factor **M** = **L Lᵀ**
2. Form the reduced matrix **K′** = **L⁻¹ K L⁻ᵀ** (standard symmetric problem)
3. Solve **K′ Ψ** = **Ψ Λ** with `numpy.linalg.eigh` (ascending order)
4. Back-transform **Φ** = **L⁻ᵀ Ψ**  →  M-orthonormal mode shapes

---

## Stiffness matrix strategies

| Strategy | When to use |
|---|---|
| `diagonal_stiffness(n, k_values)` | Joint-spring approximation; suitable for rigid robots with known compliance |
| `stiffness_from_gravity_hessian(fn, q_eq)` | Numerical linearisation of gravity about equilibrium; no URDF stiffness data needed |

---

## Modal coordinates

| Signal | Physical space | Modal space | Formula |
|---|---|---|---|
| Displacement | q ∈ ℝⁿ | η ∈ ℝᵏ | η = Φᵀ M q |
| Velocity | v ∈ ℝⁿ | η̇ ∈ ℝᵏ | η̇ = Φᵀ M v |
| Momentum | p = Mv ∈ ℝⁿ | μ ∈ ℝᵏ | μ = Φᵀ p |

Reconstruction (decode):

| Formula | Interpretation |
|---|---|
| q ≈ Φ η | Rank-*k* approximation in M-norm |
| p = M Φ μ | Exact when all modes are retained |

---

## Basis switching

Different locomotion phases (walking, jumping, push-recovery) activate
different mode shapes.  Three helpers are provided:

* **`select_modes(basis, indices)`** — arbitrary subset by column index.
* **`select_low_frequency_modes(basis, k)`** — retain the *k* lowest-ω modes.
* **`select_modes_by_frequency_range(basis, ω_min, ω_max)`** — band-pass selection.

---

## Implementation

All types and functions are importable directly from `hamiltonian_modal.modal`:

```python
from hamiltonian_modal.modal import (
    compute_modal_basis,
    diagonal_stiffness,
    encode_position,
    decode_position,
    select_low_frequency_modes,
)
```

