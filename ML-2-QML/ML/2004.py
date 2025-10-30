"""Enhanced QCNN model with residual connections and regularization."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Linear → BatchNorm → ReLU → Linear → Add residual."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        return self.relu(x + residual)


class QCNNEnhanced(nn.Module):
    """Stacked feature map, residual convolutional layers, pooling, and classification."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            ResidualBlock(32, 32),
            nn.Dropout(p=0.2),
        )
        self.pool1 = nn.AdaptiveAvgPool1d(16)

        self.conv2 = nn.Sequential(
            ResidualBlock(16, 16),
            nn.Dropout(p=0.2),
        )
        self.pool2 = nn.AdaptiveAvgPool1d(8)

        self.conv3 = nn.Sequential(
            ResidualBlock(8, 8),
            nn.Dropout(p=0.2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x.unsqueeze(0)).squeeze(0)
        x = self.conv2(x)
        x = self.pool2(x.unsqueeze(0)).squeeze(0)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNEnhanced:
    """Factory returning a fully configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()


__all__ = ["QCNN", "QCNNEnhanced"]
