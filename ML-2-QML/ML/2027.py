"""
Module: qcnn_enhanced_ml
Provides an enhanced classical QCNN with residual blocks, batch‑norm, and dropout.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class QCNNEnhanced(nn.Module):
    """
    A fully‑connected network that emulates a quantum convolutional neural network.
    Features:
    * Residual connections between layers.
    * Batch‑normalisation after every linear transformation.
    * Dropout after pooling to mitigate over‑fitting.
    * Configurable hidden sizes for experimentation.
    """

    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: tuple[int, int, int] = (16, 12, 8),
                 dropout: float = 0.1,
                 seed: Optional[int] = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )

        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )

        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
        )

        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
        )

        self.head = nn.Linear(hidden_dims[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature map
        fx = self.feature_map(x)

        # First conv + residual
        cx = self.conv1(fx)
        cx = cx + fx  # residual

        # Pool1 + residual
        px = self.pool1(cx)
        px = px + cx  # residual

        # Conv2 + residual
        cx2 = self.conv2(px)
        cx2 = cx2 + px  # residual

        # Pool2 + residual
        px2 = self.pool2(cx2)
        px2 = px2 + cx2  # residual

        # Conv3 + residual
        cx3 = self.conv3(px2)
        cx3 = cx3 + px2  # residual

        return torch.sigmoid(self.head(cx3))


def QCNNEnhancedFactory(seed: Optional[int] = None) -> QCNNEnhanced:
    """
    Factory returning a configured :class:`QCNNEnhanced` instance.
    """
    return QCNNEnhanced(seed=seed)


__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
