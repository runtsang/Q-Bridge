"""Hybrid classical QCNN with a fully‑connected layer.

This module defines a neural network that emulates the quantum convolution
and pooling operations with fully connected layers in a classical
implementation.  It extends the original QCNNModel by adding a
parameterized fully‑connected block that mirrors the quantum
fully‑connected layer from the FCL example.  The design allows
easy comparison of classical approximations of the quantum architecture
and serves as a baseline for further hybrid training.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable

class QCNNFCLModel(nn.Module):
    """Classical network that mimics a QCNN followed by a fully‑connected layer."""
    
    def __init__(self, n_features: int = 8) -> None:
        super().__init__()
        # Feature map emulation
        self.feature_map = nn.Sequential(nn.Linear(n_features, 16), nn.Tanh())
        # Convolution‑pooling blocks
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        # Fully connected block (mimics quantum FCL)
        self.fully = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fully(x)
        return torch.sigmoid(x)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Run the fully‑connected block on a sequence of parameters."""
        thetas_tensor = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.fully(thetas_tensor)).mean()

def QCNNFCL() -> QCNNFCLModel:
    """Factory returning a fully configured QCNN‑FCL model."""
    return QCNNFCLModel()

__all__ = ["QCNNFCL", "QCNNFCLModel"]
