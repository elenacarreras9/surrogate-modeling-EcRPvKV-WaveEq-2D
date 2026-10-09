"""
Surrogate model architecture (v1: MLP on a reduced quantity).

v1: predicts a scalar or a curve derived from the simulation, from the
input parameters (e.g. source amplitude, bubble number density, medium
properties).

v2 (optional, future work): if the surrogate is extended to predict the full
space-time field, this architecture would change to a CNN or to an operator
network such as FNO/DeepONet. Not implemented.
"""

import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    """Simple MLP: input parameters -> target quantity.

    Args:
        input_dim: number of physical input parameters.
        output_dim: dimension of the output (1 for a scalar, larger for a
            discretized curve).
        hidden_dim: size of the hidden layers.
        n_hidden_layers: number of hidden layers.
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
