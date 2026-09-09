"""
Evaluacion del surrogate: error frente al solver y comparacion de tiempos
de inferencia vs. tiempo de simulacion (Fase 3 del plan).

Pendiente de completar cuando haya un modelo entrenado (train.py) y un
conjunto de test separado.
"""

import time

import numpy as np
import torch


def relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Error relativo (MAE y RMSE) entre la prediccion del surrogate y el solver."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"mae": mae, "rmse": rmse}


def benchmark_inference_time(model: torch.nn.Module, x_sample: torch.Tensor, n_runs: int = 100) -> float:
    """Tiempo medio de inferencia del surrogate, para comparar con el tiempo
    de una simulacion equivalente en MATLAB (argumento de "aceleracion").
    """
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_runs):
            model(x_sample)
        elapsed = time.perf_counter() - start
    return elapsed / n_runs


if __name__ == "__main__":
    raise NotImplementedError("Pendiente: cargar modelo entrenado y dataset de test (Fase 3)")
