"""Classical convolution‑inspired network with residuals and regularisation.

This module provides a lightweight CNN that mirrors the structure of the
original QCNN but introduces residual connections, dropout and
batch‑normalisation to improve generalisation and training stability.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Single convolution‑pooling block with optional skip connection."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
        )
        self.pool = nn.Sequential(
            nn.Linear(out_features, out_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_features // 2),
            nn.Dropout(dropout),
        )
        self.skip = nn.Linear(in_features, out_features // 2) if in_features!= out_features // 2 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.pool(y)
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(y + x) if self.skip is not None else y


class QCNNEnhanced(nn.Module):
    """Stack of ConvBlocks emulating a quantum convolution network with residuals."""
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        self.conv1 = ConvBlock(16, 16, dropout)
        self.conv2 = ConvBlock(8, 8, dropout)  # after pooling, dim halves to 8
        self.conv3 = ConvBlock(4, 4, dropout)   # after pooling, dim halves to 4
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def evaluate(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Evaluate the network without training‑time dropout."""
        self.eval()
        if device is not None:
            x = x.to(device)
        with torch.no_grad():
            return self.forward(x)

def QCNNEnhancedFactory(dropout: float = 0.2) -> QCNNEnhanced:
    """Convenience factory returning a freshly initialised QCNNEnhanced."""
    return QCNNEnhanced(dropout=dropout)

__all__ = ["QCNNEnhanced", "QCNNEnhancedFactory"]
