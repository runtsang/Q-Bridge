"""Enhanced classical fully connected layer with two hidden layers and dropout."""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """
    A two‑layer feed‑forward network that mimics a quantum fully‑connected layer.
    Parameters are learned in a standard PyTorch training loop.
    The public ``run`` method keeps the same signature as the original seed,
    accepting a sequence of floats that represent input features and returning
    a single‑valued numpy array, suitable for direct comparison with the QML
    implementation.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run a forward pass on a single example.
        Args:
            thetas: Iterable of input features.
        Returns:
            A one‑element numpy array with the network output.
        """
        x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.forward(x)
        # Mean over batch dimension to match original behaviour
        expectation = out.mean(dim=0).detach().numpy()
        return expectation

__all__ = ["FCL"]
