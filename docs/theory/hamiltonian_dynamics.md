# Hamiltonian Dynamics Theory

## Overview

The Hamiltonian world model predicts modal-state evolution via Hamilton's equations, numerically integrated with a symplectic scheme.

## Hamiltonian Decomposition

$$H_\theta(\eta, \dot{\eta}, m) = T(\dot{\eta}) + V_\theta(\eta, m) + \Delta_{\text{contact},\theta}(\eta, m)$$

Where:
- $T(\dot{\eta}) = \frac{1}{2}\|\dot{\eta}\|^2$ — kinetic energy (analytical; mass-orthonormal modes)
- $V_\theta(\eta, m)$ — learned potential energy MLP, mode-conditioned
- $\Delta_{\text{contact},\theta}(\eta, m)$ — learned contact correction MLP

## Hamilton's Equations

$$\dot{\eta} = \frac{\partial H}{\partial \dot{\eta}} = \dot{\eta}, \qquad \ddot{\eta} = -\frac{\partial H}{\partial \eta} = -\frac{\partial V_\theta}{\partial \eta} - \frac{\partial \Delta_\theta}{\partial \eta}$$

## Contact Mode Conditioning

The contact mode $m \in \{0,1,2,3,4\}$ is one-hot encoded and concatenated with $\eta$ as input to $V_\theta$ and $\Delta_\theta$:

$$x = [\eta; \text{one-hot}(m)] \in \mathbb{R}^{r + n_m}$$

## Architecture

- **$V_\theta$**: 4-layer MLP, $[r + n_m, 256, 256, 256, 256, 1]$, tanh activations
- **$\Delta_\theta$**: 3-layer MLP, $[r + n_m, 128, 128, 128, 1]$, tanh activations

## Training Loss

$$\mathcal{L} = \text{MSE}(\eta_\text{pred}, \eta_\text{true}) + \lambda_Q \mathcal{L}_\text{quantile} + \lambda_C \mathcal{L}_\text{contact} + \lambda_E \mathcal{L}_\text{energy}$$

where $\mathcal{L}_\text{energy} = (H(\eta_{t+1}, \dot{\eta}_{t+1}, m) - H(\eta_t, \dot{\eta}_t, m))^2$.
