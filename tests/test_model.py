"""
Basic tests of the model. Run with `pytest` from the repository root
(with the virtual environment activated).

They check that the input/output shapes are the expected ones before relying
on the training "seeming" to work.
"""

import numpy as np
import torch

from src.evaluate import error_metrics
from src.model import SurrogateMLP


def test_output_shape():
    model = SurrogateMLP(input_dim=5, output_dim=1)
    x = torch.randn(8, 5)  # batch of 8 samples, 5 parameters each
    y = model(x)
    assert y.shape == (8, 1)


def test_forward_does_not_crash_with_different_batch_sizes():
    model = SurrogateMLP(input_dim=3, output_dim=2)
    for batch_size in (1, 4, 16):
        x = torch.randn(batch_size, 3)
        y = model(x)
        assert y.shape == (batch_size, 2)


def test_error_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 5.0])  # errors: 0, 1, 2
    m = error_metrics(y_true, y_pred)
    assert np.isclose(m["mae"], 1.0)
    assert np.isclose(m["rmse"], np.sqrt(5.0 / 3.0))
