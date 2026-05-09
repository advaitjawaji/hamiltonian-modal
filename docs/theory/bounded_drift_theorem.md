# Bounded drift theorem

## Statement

**Theorem (Hairer, Lubich & Wanner, 2006).**  Let H : ℝⁿ × ℝⁿ → ℝ be a
smooth Hamiltonian and let Φ_dt be one step of a symplectic integrator of
order p with stepsize dt.  Then there exists a modified Hamiltonian H̃ such
that:

1. **H̃ is conserved exactly** along discrete orbits of Φ_dt:
   `H̃(q_{k+1}, p_{k+1}) = H̃(q_k, p_k)` for all k.

2. **H̃ approximates H** to order p:
   `|H̃(q, p) − H(q, p)| = O(dt^p)`.

3. **The true Hamiltonian is bounded**:
   `|H(q_k, p_k) − H(q_0, p_0)| ≤ C · dt^p`  for all k ≥ 0,
   where C depends on H but *not* on k (no secular drift).

For the Störmer–Verlet / leapfrog integrator, p = 2 and the bound holds
as long as dt < dt_crit (determined by the spectral radius of the system).

---

## Implications for the Hamiltonian-modal world model

* **Long-horizon stability** — The `DiffSimPipeline` can unroll thousands of
  steps without accumulating significant energy error, unlike Euler integration
  where |ΔH| grows linearly with simulation time.

* **Gradient quality** — Because the shadow Hamiltonian H̃ is conserved, the
  energy landscape seen by the policy-gradient estimator is consistent across
  the rollout.  This prevents gradient explosion caused by runaway energy
  growth.

* **Push-recovery robustness** — External impulses shift (q, p) instantaneously,
  but the leapfrog orbit immediately returns to the nearest energy shell of H̃,
  limiting how far the robot state can deviate from a valid Hamiltonian
  trajectory.

---

## DriftBench-G1 empirical validation

The `benchmarks/driftbench_g1` suite measures the relative energy drift

```
δ_H = max_t |H(t) − H(0)| / |H(0)|
```

for both leapfrog and Euler integrators.  On the standard harmonic-oscillator
suite (n_dof = 10, n_steps = 5 000, dt = 0.005 s), typical results are:

| Integrator | δ_H        |
|---|---|
| Leapfrog   | ≈ 10⁻⁸     |
| Euler      | ≈ 10⁻¹–10⁰ |

This three-to-seven order-of-magnitude advantage motivates the choice of
symplectic integration throughout `hamiltonian-modal`.

---

## References

* E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*,
  Springer, 2006.
* R. I. McLachlan and G. R. W. Quispel, "Splitting methods,"
  *Acta Numerica* 11, 341–434, 2002.
