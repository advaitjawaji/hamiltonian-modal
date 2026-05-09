# Contributing to hamiltonian-modal

Thank you for your interest in contributing! Please follow these guidelines.

## Fork and Branch

1. Fork the repository on GitHub.
2. Clone your fork locally.
3. Create a branch using the naming convention:

   ```
   feat/task-X.Y-short-description
   ```

   Examples: `feat/task-1.2-modal-decomp`, `feat/task-2.1-world-model-train`

## Setting Up

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Pre-commit

All commits must pass pre-commit hooks (black, ruff, mypy). Run manually with:

```bash
pre-commit run --all-files
```

## Running Tests

```bash
pytest                       # run all tests
pytest tests/unit/           # unit tests only
pytest -m "not slow"         # skip slow tests
pytest --cov=hamiltonian_modal --cov-report=term-missing
```

## Pull Request Format

- Title: `feat: <short description>` / `fix: <short description>` / `docs: <short description>`
- Body must include:
  - **Summary** — what the PR does and why
  - **Test plan** — which tests cover the change
  - **Related issues** — link any relevant GitHub issues
- All CI checks must pass before merge.
- At least one maintainer review is required.
