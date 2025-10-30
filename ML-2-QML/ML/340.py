"""Extended classical QCNN with residuals, batch‑norm and dropout."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Simple residual block: Linear → BatchNorm → Tanh → dropout → Linear → addition."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        # match dimensions if needed
        if residual.shape[-1]!= out.shape[-1]:
            residual = nn.functional.pad(residual, (0, out.shape[-1] - residual.shape[-1]))
        return out + residual


class QCNNExtendedModel(nn.Module):
    """Convolution‑inspired network with residual blocks and dropout."""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = ResidualBlock(8, 16, dropout)
        self.conv1 = ResidualBlock(16, 16, dropout)
        self.pool1 = ResidualBlock(16, 12, dropout)
        self.conv2 = ResidualBlock(12, 8, dropout)
        self.pool2 = ResidualBlock(8, 4, dropout)
        self.conv3 = ResidualBlock(4, 4, dropout)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNNExtended(dropout: float = 0.1) -> QCNNExtendedModel:
    """Factory returning the configured :class:`QCNNExtendedModel`."""
    return QCNNExtendedModel(dropout=dropout)


__all__ = ["QCNNExtended", "QCNNExtendedModel"]
