"""Enhanced fully connected layer using a multi‑layer perceptron.

The class exposes a ``run`` method that accepts an iterable of input
features, applies a two‑layer neural network with ReLU non‑linearity,
dropout and a final tanh activation, and returns the mean output as a
NumPy array.  The implementation is fully torch‑based and can be used
for both inference and training.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def FCL():
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1, hidden_dim: int = 32, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),
            )

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Forward pass on the provided input features.

            Parameters
            ----------
            thetas : Iterable[float]
                Iterable of input feature values.

            Returns
            -------
            np.ndarray
                Mean of the network output as a 1‑D NumPy array.
            """
            inputs = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            with torch.no_grad():
                output = self.net(inputs)
            return output.mean(dim=0).detach().numpy()

    return FullyConnectedLayer()


__all__ = ["FCL"]
