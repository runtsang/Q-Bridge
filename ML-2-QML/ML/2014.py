"""Enhanced classical QCNN model with residual connections and dropout.

This module defines a deeper, regularised neural network that mimics the
structure of a quantum convolutional neural network (QCNN).  The architecture
uses residual blocks, batch‑normalisation and dropout to improve generalisation
for small datasets typical of quantum benchmark problems.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNModel(nn.Module):
    """
    A deeper, regularised QCNN-inspired neural network.

    The architecture expands the baseline linear layers into a series of
    residual blocks that mimic quantum convolution and pooling stages.
    Dropout and batch‑normalisation are interleaved to improve generalisation
    for small datasets commonly used in quantum benchmarking.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: int = 16) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU()
        )

        # Convolution‑like residual block 1
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Pooling‑like transformation 1
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.BatchNorm1d(hidden_dims // 2),
            nn.ReLU()
        )

        # Convolution‑like residual block 2
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dims // 2, hidden_dims // 2),
            nn.BatchNorm1d(hidden_dims // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Pooling‑like transformation 2
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims // 2, hidden_dims // 4),
            nn.BatchNorm1d(hidden_dims // 4),
            nn.ReLU()
        )

        # Final head
        self.head = nn.Linear(hidden_dims // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        # Residual block 1
        res = self.res1(x)
        x = torch.relu(x + res)
        # Pool 1
        x = self.pool1(x)
        # Residual block 2
        res = self.res2(x)
        x = torch.relu(x + res)
        # Pool 2
        x = self.pool2(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
