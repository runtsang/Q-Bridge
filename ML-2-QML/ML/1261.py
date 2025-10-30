"""
QCNNModel – an enhanced classical convolution‑inspired network.

Features
--------
* Residual blocks that bypass the convolutional layers to mitigate vanishing gradients.
* Batch‑normalisation after each linear transformation to stabilise training.
* Dropout for regularisation.
* Flexible input dimension – the first layer adapts to the feature size of the dataset.
* A `feature_extractor` method that returns the representation before the final head,
  useful for downstream tasks or transfer learning.

Design
------
The architecture mirrors the original seed but replaces the simple linear stacks with
`ResidualBlock`s that contain a linear transform followed by a ReLU, batch‑norm and
dropout, then add the input back.  The pool layers are implemented as simple linear
reductions with a ReLU.  The final head maps to a single output (sigmoid for binary
classification).

The class is fully compatible with PyTorch’s `nn.Module` API and can be dropped
into any training loop or integrated with PyTorch Lightning.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Callable


class ResidualBlock(nn.Module):
    """Linear residual block with batch‑norm and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.residual = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.residual(x)


class QCNNModel(nn.Module):
    """Enhanced QCNN‑style feed‑forward network."""

    def __init__(self, input_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        # Feature map
        self.feature_map = ResidualBlock(input_dim, 16, dropout)

        # Convolutional stages
        self.conv1 = ResidualBlock(16, 16, dropout)
        self.conv2 = ResidualBlock(16, 12, dropout)
        self.conv3 = ResidualBlock(12, 8, dropout)

        # Pooling stages (simple linear reduction)
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.pool3 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

        # Final head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return torch.sigmoid(self.head(x))

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Return the representation before the final head."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        return x


def QCNN() -> QCNNModel:
    """Factory returning a pre‑configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
