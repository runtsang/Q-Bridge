"""Hybrid classical QCNN implementation with depth, BN, and dropout.

The model mirrors the original QCNN helper but adds modern
neural‑network best practices: batch normalization, ReLU
activations, and dropout for regularisation.  The architecture
is still fully connected, making it easy to plug into standard
PyTorch pipelines.

Author: gpt-oss-20b
"""

from __future__ import annotations
from typing import Iterable

import torch
from torch import nn


class QCNNModel(nn.Module):
    """Classical QCNN‑style network with modern layers.

    The network processes 8‑dimensional inputs through a
    feature map and three convolution‑pool stages that mimic the
    QCNN's layered approach.  Each stage includes a linear
    transformation, batch‑norm, ReLU, and dropout for
    regularisation.  The final head maps the 4‑dimensional
    representation to a scalar output via a sigmoid.
    """

    def __init__(self, dropout: float = 0.1, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Convolution‑pool stage 1
        self.conv1 = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Convolution‑pool stage 2
        self.conv2 = nn.Sequential(
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Convolution‑pool stage 3
        self.conv3 = nn.Sequential(
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning a fully‑configured QCNNModel.

    The returned model is ready for training with any
    PyTorch optimiser.
    """
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
