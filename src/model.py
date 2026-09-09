"""
Arquitectura del surrogate model (v1: MLP sobre magnitud reducida).

v1: predice una cantidad escalar/curva derivada de la simulacion a partir de
los parametros de entrada (frecuencia, propiedades del medio, radio de
burbuja, etc.) -- ver el plan del proyecto para el detalle de la eleccion.

v2 (stretch, opcional): si se amplia a predecir el campo completo
espacio-temporal, esta arquitectura cambiaria a una CNN o a un operador tipo
FNO/DeepONet -- no implementado todavia.
"""

import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    """MLP simple: parametros de entrada -> magnitud objetivo.

    Args:
        input_dim: numero de parametros fisicos de entrada.
        output_dim: dimension de la salida (1 si es un escalar,
            mayor si es una curva discretizada).
        hidden_dim: tamano de las capas ocultas.
        n_hidden_layers: numero de capas ocultas.
    """

    def __init__(self, input_dim: int, output_dim: int = 1,
                 hidden_dim: int = 64, n_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
