# Hamiltonian dynamics

## Lagrangian mechanics and the Hamiltonian formulation

For a mechanical system with generalised coordinates **q** ∈ ℝⁿ and velocities
**q̇**, the Lagrangian is

```
L(q, q̇) = T(q, q̇) − V(q)
```

where **T** is kinetic energy and **V** is potential energy.  Passing to
generalised momenta **p** = ∂L/∂q̇ = **M**(q) **q̇** yields the Hamiltonian

```
H(q, p) = pᵀ q̇ − L = T(p, q) + V(q)
```

Hamilton's equations of motion are:

```
q̇ =  ∂H/∂p = M(q)⁻¹ p
ṗ = −∂H/∂q = −∂V/∂q + (inertia terms)
```

For our separable approximation H = T(p) + V(q) (diagonal in modal space):

```
η̇ = ∂H/∂μ = M_modal⁻¹ μ
μ̇ = −∂H/∂η = −∂V/∂η
```

where **η** are modal coordinates and **μ** are modal momenta (see
[Modal decomposition](modal_decomposition.md)).

---

## Hamiltonian neural network

`HamiltonianNet` parameterises the total energy as:

```
H(η, μ) = T(μ) + V(η)
```

**Kinetic term** T(μ) = ½ μᵀ (L Lᵀ)⁻¹ μ

where **L** is a learned lower-triangular Cholesky factor.  This guarantees
T > 0 and preserves the positive-definite structure of the mass matrix.

**Potential term** V(η) = softplus(MLP(η))

A small tanh-MLP followed by a softplus activation forces V ≥ 0 everywhere,
consistent with a stable equilibrium at the origin.

---

## Gradient computation

Because `hamiltonian-modal` uses pure NumPy, gradients are computed via
central finite differences:

```
∂H/∂ηᵢ ≈ [H(η + εeᵢ, μ) − H(η − εeᵢ, μ)] / (2ε)
```

The kinetic gradient ∂H/∂μ = M_modal⁻¹ μ is computed exactly (linear solve).

---

## Energy conservation

For an exact Hamiltonian system, **H** is a constant of motion.  Discretised
systems only conserve a *shadow Hamiltonian* H̃ = H + O(dt²).  Symplectic
integrators guarantee that H̃ is *exactly* conserved, bounding the drift of
H itself — see [Symplectic integration](symplectic_integration.md).
