"""Hybrid QCNN model combining classical and optional quantum components."""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable, Optional

class QCNNHybrid(nn.Module):
    """
    Hybrid QCNN that emulates the quantum convolutional architecture
    with fully‑connected layers and can inject a quantum sub‑module
    via a callable weight function.
    """
    def __init__(
        self,
        input_dim: int = 8,
        quantum_submodule: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.quantum_submodule = quantum_submodule

        # Classical emulation of QCNN layers
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If a quantum submodule is provided, apply it first
        if self.quantum_submodule is not None:
            x = self.quantum_submodule(x)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNNHybridFactory(
    quantum_submodule: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> QCNNHybrid:
    """
    Factory that returns a fully configured QCNNHybrid instance.
    """
    return QCNNHybrid(quantum_submodule=quantum_submodule)

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
