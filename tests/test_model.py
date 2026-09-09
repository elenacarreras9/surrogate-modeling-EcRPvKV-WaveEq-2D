"""
Tests basicos del modelo. Ejecutar con `pytest` desde la raiz del proyecto
(con el venv activado).

Sirven como ejemplo minimo de buenas practicas: comprobar que las formas de
entrada/salida son las esperadas antes de fiarse de que el entrenamiento
"parece" funcionar.
"""

import torch

from src.model import SurrogateMLP


def test_output_shape():
    model = SurrogateMLP(input_dim=5, output_dim=1)
    x = torch.randn(8, 5)  # batch de 8 muestras, 5 parametros cada una
    y = model(x)
    assert y.shape == (8, 1)


def test_forward_does_not_crash_with_different_batch_sizes():
    model = SurrogateMLP(input_dim=3, output_dim=2)
    for batch_size in (1, 4, 16):
        x = torch.randn(batch_size, 3)
        y = model(x)
        assert y.shape == (batch_size, 2)
