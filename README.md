# Surrogate modeling of nonlinear ultrasound propagation in bubbly viscoelastic media (2D)

Scaffolding for a machine-learning surrogate of a 2D wave-propagation model for bubbly viscoelastic media (Kelvin–Voigt matrix with gas bubbles described by a volume-based Rayleigh–Plesset equation).

> **Status: work in progress. This repository contains project scaffolding only.**
> It contains no data, no trained models and no results. The MATLAB 2D solver that would generate the training data is **not** included in this repository.

## Goal

The 2D finite-difference solver is accurate but expensive. The aim is to train a neural-network surrogate that approximates its output for a given set of physical parameters, so that parameter sweeps and inverse problems become much cheaper than running the solver every time.

Planned work:

1. Run a parameter sweep over the source amplitude `p0` and the initial gas content `Ng0` in a viscoelastic example medium.
2. Define the target magnitude the surrogate predicts (still being decided).
3. Train a baseline MLP surrogate and evaluate it on held-out simulations.
4. Future work: physics-informed neural networks (PINNs).

## Relation to the 1D project

A complete, working end-to-end example of the same idea (dataset generation, MLP training, evaluation, prediction and inversion) is available for the 1D case:
[surrogate-modeling-bubbly-viscoelastic-1d](https://github.com/elenacarreras9/surrogate-modeling-bubbly-viscoelastic-1d).
This 2D repository follows the same structure but is at a much earlier stage.

## Repository layout

```
src/
  model.py             SurrogateMLP: fully connected network (ReLU)
  generate_dataset.py  Skeleton for building the dataset (not implemented)
  train.py             Training loop (dataset loading not implemented)
  evaluate.py          Error metrics (MAE, RMSE) and inference-time benchmark
tests/
  test_model.py        Unit tests for the model and the metrics
data/
  raw/                 Raw simulation outputs (empty, not tracked)
  processed/           Processed datasets (empty, not tracked)
notebooks/             Exploration notebooks (empty)
```

## Current state

| Component | State |
|---|---|
| MLP model (`src/model.py`) | Implemented and unit-tested |
| Error metrics and timing (`src/evaluate.py`) | Implemented and unit-tested |
| Training loop (`src/train.py`) | Implemented; dataset loading pending |
| Dataset generation (`src/generate_dataset.py`) | Skeleton only |
| 2D solver | Not included in this repository |
| Data, trained models, results | None yet |

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## License

MIT, see [LICENSE](LICENSE).
