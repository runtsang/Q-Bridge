from __future__ import annotations

import torch
from torch import nn

class HybridQCNNModel(nn.Module):
    """Classical neural network inspired by the QCNN architecture with residual and dropout enhancements.

    The model emulates the quantum convolution‑pooling stages using fully‑connected
    layers and introduces skip connections to mitigate vanishing gradients.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
        )
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
        )
        # Residual connection
        self.residual = nn.Linear(4, 4)
        # Regression head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # residual addition
        x = x + self.residual(x)
        return torch.sigmoid(self.head(x))

def HybridQCNN() -> HybridQCNNModel:
    """Factory that returns a fully‑configured :class:`HybridQCNNModel`."""
    return HybridQCNNModel()

__all__ = ["HybridQCNN", "HybridQCNNModel"]
