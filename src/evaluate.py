"""
Evaluation of the surrogate: error with respect to the solver, and comparison
of the inference time with the simulation time.

To be completed once a trained model (train.py) and a separate test set exist.
"""

import time

import numpy as np
import torch


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Absolute error metrics (MAE and RMSE, in the units of the target)
    between the surrogate prediction and the solver result."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def benchmark_inference_time(model: torch.nn.Module, x_sample: torch.Tensor, n_runs: int = 100) -> float:
    """Mean inference time of the surrogate, to compare with the time of an
    equivalent solver simulation (the "speed-up" argument).
    """
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x_sample)
        elapsed = time.perf_counter() - start
    return elapsed / n_runs


if __name__ == "__main__":
    raise NotImplementedError("Pending: load a trained model and a test dataset")
