# hamiltonian-modal

`hamiltonian-modal` is a Python package scaffold for Hamiltonian-modal world models,
modal dynamics, and long-horizon planning experiments on humanoid robots such as
Unitree G1.

## Phase 1 scaffold

This repository currently provides the initial package and documentation layout for:

- modal decomposition and projection utilities
- Hamiltonian world-model components
- planning, policy, MPC, and differentiable simulation modules
- environment, data, benchmarking, and baseline integration surfaces
- Pinocchio-based G1 model loading and CRBA mass-matrix evaluation

## Development

Install the package in editable mode and run the test suite:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```
