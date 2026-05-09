# Modal Decomposition Theory

## Overview

The modal decomposition projects the joint-coordinate state $(q, \dot{q})$ of the Unitree G1 into modal coordinates $(\eta, \dot{\eta})$ via mode shapes computed from the system's mass matrix $M$ and stiffness matrix $K$.

## Generalized Eigenvalue Problem

Mode shapes $\Phi \in \mathbb{R}^{n \times r}$ and squared natural frequencies $\omega_i^2$ satisfy:

$$K \Phi = M \Phi \, \text{diag}(\omega_1^2, \ldots, \omega_r^2)$$

Equivalently, $(K - \omega_i^2 M)\phi_i = 0$ for each mode $i$.

**Scipy implementation:** `scipy.linalg.eigh(K, M)` solves this directly, returning mass-orthonormal eigenvectors.

## Mass Orthonormality

The mode shapes satisfy:

$$\Phi^T M \Phi = I_r, \qquad \Phi^T K \Phi = \text{diag}(\omega_1^2, \ldots, \omega_r^2)$$

## Modal Encoding

$$\eta = \Phi^T M (q - q_\text{ref}), \qquad \dot{\eta} = \Phi^T M \dot{q}$$

## Modal Decoding

$$q = q_\text{ref} + \Phi \eta, \qquad \dot{q} = \Phi \dot{\eta}$$

## G1 Stiffness Matrix

The effective stiffness $K$ combines:
1. **Joint PD gains:** $K_\text{joint} = \text{diag}(k_1, \ldots, k_n)$, typical $k_i = 200$ N·m/rad
2. **Gearbox compliance:** $K_\text{gear} = \frac{k_\text{gearbox}}{r_\text{gear}} I$, typical $k_\text{gearbox} = 5000$ N·m/rad
3. **Link bending:** $K_\text{bend} = \frac{EI}{L^3} I$ from Euler-Bernoulli beam theory

## References

- Hoshyari et al. "Vibration-Minimizing Motion Retargeting." SIGGRAPH 2019.
- Shabana, A. "Dynamics of Multibody Systems." Cambridge University Press, 2005.
