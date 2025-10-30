from __future__ import annotations

import torch
from torch import nn

class HybridQCNN(nn.Module):
    """Classical hybrid‑QCNN network.

    The architecture is a depth‑controlled feed‑forward stack that
    emulates the convolutional blocks of the quantum circuit while
    preserving the flexibility of a standard neural net.  Each depth
    layer consists of a linear projection followed by ReLU, mirroring
    the parameter count of the quantum ansatz (see QuantumClassifierModel).
    """
    def __init__(self, num_features: int, depth: int = 3, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or num_features
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))  # binary classification
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

def HybridQCNNFactory(num_features: int, depth: int = 3, hidden_dim: int | None = None) -> HybridQCNN:
    """Return a configured instance of :class:`HybridQCNN`."""
    return HybridQCNN(num_features, depth, hidden_dim)

__all__ = ["HybridQCNN", "HybridQCNNFactory"]
