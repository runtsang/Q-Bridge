from __future__ import annotations

from typing import Iterable
import numpy as np
import torch
from torch import nn


class HybridFCLQCNN(nn.Module):
    """
    Classical hybrid network that blends the FCL linear layer with QCNN‑style
    convolution and pooling blocks.  The architecture follows:
      1. Feature extraction: Linear → Tanh
      2. Convolution 1: Linear → Tanh
      3. Pooling 1: Linear → Tanh (dim reduction)
      4. Convolution 2: Linear → Tanh
      5. Pooling 2: Linear → Tanh (dim reduction)
      6. Final fully‑connected output
    The forward pass accepts a 2‑D tensor of shape (batch, features).
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden_dim // 4, hidden_dim // 8), nn.Tanh())
        self.fc = nn.Linear(hidden_dim // 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.fc(x)
        return torch.sigmoid(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the interface of the original FCL module.  ``thetas`` is a flat
        iterable of floats that will be reshaped into a column vector and fed
        through the network.  The output is a NumPy array of the predicted
        probabilities.
        """
        thetas_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        prob = self.forward(thetas_tensor)
        return prob.detach().numpy()


def HybridFCLQCNNFactory() -> HybridFCLQCNN:
    """Return a ready‑to‑use instance of the hybrid network."""
    return HybridFCLQCNN()


__all__ = ["HybridFCLQCNN", "HybridFCLQCNNFactory"]
