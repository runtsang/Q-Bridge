"""Hybrid QCNN model combining classical convolutional layers with a classifier head.

The architecture mirrors the quantum QCNN structure but uses fully‑connected layers.
It can be trained jointly with the quantum model via weight sharing or used as a baseline.
"""

import torch
from torch import nn

class QCNNHybrid(nn.Module):
    """
    Classical analogue of a quantum convolutional neural network with a classifier head.

    Architecture:
    - Feature map: 8 → 16
    - Conv1: 16 → 16
    - Pool1: 16 → 12
    - Conv2: 12 → 8
    - Pool2: 8 → 4
    - Conv3: 4 → 4
    - Head: 4 → 2 (binary classification)

    All activations are Tanh except the final layer which uses Sigmoid for probability.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def weight_sizes(self) -> list[int]:
        """Return a list of the number of parameters per linear layer."""
        return [p.numel() for p in self.parameters()]

__all__ = ["QCNNHybrid"]
