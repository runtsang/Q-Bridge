"""Classical QCNN implementation with residual connections and a deeper feature extractor."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class QCNNModel(nn.Module):
    """
    A deep convolution‑inspired network that mirrors the structure of the original QCNN,
    but with residual skip connections and a richer feature extractor.

    The architecture:
        - Feature map: Linear(8→32) + ReLU
        - Conv1: Linear(32→32) + ReLU
        - Pool1: Linear(32→24) + ReLU
        - Conv2: Linear(24→24) + ReLU
        - Pool2: Linear(24→16) + ReLU
        - Conv3: Linear(16→16) + ReLU
        - Residual: Add input of Conv1 to output of Conv3
        - Head: Linear(16→1) + Sigmoid

    The forward pass returns a probability in [0,1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(24, 24),
            nn.ReLU()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.head = nn.Linear(16, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Residual connection from conv1 output
        x = x + x1
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """
    Factory returning a fully configured QCNNModel instance.
    """
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
