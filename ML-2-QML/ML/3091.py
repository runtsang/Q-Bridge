"""Hybrid fully‑connected layer: classical implementation.

Provides a feed‑forward neural network that mimics the quantum‑parameterised circuit.
The class exposes a `run` method that accepts an iterable of input values and returns a
single‑dimensional numpy array of predictions.  The implementation is fully
compatible with PyTorch and can be trained with standard optimisers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List

class HybridFullyConnectedClassifier(nn.Module):
    """
    Classical feed‑forward network with a configurable depth.

    Parameters
    ----------
    n_features : int
        Number of input features.
    depth : int
        Number of hidden layers.
    num_classes : int
        Size of the output layer (default 2 for binary classification).
    """

    def __init__(self, n_features: int = 1, depth: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, n_features))
            layers.append(nn.ReLU())
            in_dim = n_features
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network on a single data point.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of real numbers representing the feature vector.

        Returns
        -------
        np.ndarray
            1‑D array of shape (num_classes,).
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        logits = self.forward(values)
        return logits.detach().cpu().numpy().squeeze()

def FCL() -> type[HybridFullyConnectedClassifier]:
    """Return the class that implements the classical hybrid layer."""
    return HybridFullyConnectedClassifier
