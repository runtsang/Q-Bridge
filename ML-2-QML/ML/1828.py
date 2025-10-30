"""Enhanced classical QCNN with residuals, batch‑norm and dropout."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EnhancedQCNNModel(nn.Module):
    """
    A deeper, more expressive QCNN inspired architecture.

    The network mirrors the original quantum convolution steps but
    adds:
      * Residual connections between conv layers.
      * Batch‑normalization after each conv and pooling stage.
      * Dropout for regularisation.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
            nn.BatchNorm1d(12),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
            nn.BatchNorm1d(8),
            nn.Dropout(dropout),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.BatchNorm1d(4),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.BatchNorm1d(4),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        # Residual: conv1 + input
        conv1_out = self.conv1(x)
        x = conv1_out + x
        # Residual: pool1 + conv1_out
        pool1_out = self.pool1(x)
        x = pool1_out + conv1_out
        conv2_out = self.conv2(x)
        x = conv2_out + pool1_out
        pool2_out = self.pool2(x)
        x = pool2_out + conv2_out
        conv3_out = self.conv3(x)
        x = conv3_out + pool2_out
        logits = self.head(x)
        return torch.sigmoid(logits)


def EnhancedQCNN() -> EnhancedQCNNModel:
    """Factory returning the configured :class:`EnhancedQCNNModel`."""
    return EnhancedQCNNModel()


__all__ = ["EnhancedQCNN", "EnhancedQCNNModel"]
