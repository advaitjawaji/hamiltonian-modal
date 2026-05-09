# Symplectic integration

## Motivation

Naïve Euler integration of Hamilton's equations accumulates energy error that
grows *linearly* with time.  For long-horizon planning (seconds to tens of
seconds of robot motion) this leads to diverging world-model predictions.

Symplectic integrators are structure-preserving: they exactly conserve the
*symplectic two-form* ω = dq ∧ dp, which forces them to conserve a nearby
*shadow Hamiltonian* H̃ = H + O(dtᵖ) for all time.

---

## Störmer–Verlet / leapfrog

The velocity-form leapfrog (implemented in `leapfrog_step`) for a separable
Hamiltonian H(q, p) = T(p) + V(q):

```
p_{1/2} = p_0 − (dt/2) ∂V/∂q(q_0)
q_1     = q_0 + dt ∂T/∂p(p_{1/2})
p_1     = p_{1/2} − (dt/2) ∂V/∂q(q_1)
```

**Properties:**
| Property | Value |
|---|---|
| Order | 2nd order in dt |
| Symplecticity | Exact |
| Time-reversibility | Yes |
| Energy drift | O(dt²) per step, bounded over all time |

For non-separable H (e.g. when M depends on q), one full gradient evaluation
∂H/∂q is required at each half-step.

---

## Position-form Störmer–Verlet

`stormer_verlet_step` is equivalent to leapfrog but expressed in (q, p) rather
than the velocity-form half-step p.  It is useful when the momentum update is
the computational bottleneck.

---

## Modal-space integration

In the `DiffSimPipeline`, integration is carried out in modal coordinates:

```
μ_{1/2} = μ_0 − (dt/2) ∂V/∂η(η_0)
η_1     = η_0 + dt M_modal⁻¹ μ_{1/2}
μ_1     = μ_{1/2} − (dt/2) ∂V/∂η(η_1)
```

The diagonal structure of M_modal makes the velocity update O(k) instead of
O(k²) for a full solve.

---

## Drift bound

See [Bounded drift theorem](bounded_drift_theorem.md) for a formal statement
of the shadow-Hamiltonian conservation property and its implications for
long-horizon planning stability.
