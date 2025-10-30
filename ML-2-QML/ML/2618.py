"""Hybrid QCNN model combining classical fully‑connected layers with quantum-inspired operations.

This module defines :class:`QCNNModel`, a lightweight neural network that mimics the quantum
convolution and pooling operations used in the QCNN quantum implementation.
It exposes the same public API as the original QCNN implementation while
adding introspection of parameter counts and observable mapping.

The network is fully differentiable and can be trained with any PyTorch
optimizer.  The architecture is:
    feature_map -> conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> head

Each block is a small feed‑forward network with ReLU activations.  The
parameter counts are exposed via ``weight_sizes`` and the observable indices
via ``observables``, matching the quantum counterpart.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.pool1 = nn.Sequential(nn.Linear(hidden_dim, 12), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, output_dim)

        # Store parameter counts for introspection
        self.weight_sizes = [
            *[p.numel() for p in self.feature_map.parameters()],
            *[p.numel() for p in self.conv1.parameters()],
            *[p.numel() for p in self.pool1.parameters()],
            *[p.numel() for p in self.conv2.parameters()],
            *[p.numel() for p in self.pool2.parameters()],
            *[p.numel() for p in self.conv3.parameters()],
            *[p.numel() for p in self.head.parameters()],
        ]

        # Observable mapping matching the quantum model (single output)
        self.observables = [0]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
