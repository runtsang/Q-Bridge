"""
QCNNHybrid: a classical convolution‑inspired network with
regularization and optional dropout.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNHybrid(nn.Module):
    """
    Classical QCNN‑style network.

    Parameters
    ----------
    input_dim : int, default 8
        Dimensionality of the input feature vector.
    dropout : float, default 0.1
        Dropout probability applied after the feature map.
    weight_decay : float, default 1e-4
        L2 regularization strength used during training.
    """

    def __init__(self, input_dim: int = 8, dropout: float = 0.1,
                 weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.weight_decay = weight_decay

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def l2_penalty(self) -> torch.Tensor:
        """Return L2 penalty over all trainable parameters."""
        return sum(p.pow(2).sum() for p in self.parameters())

def QCNNHybrid() -> QCNNHybrid:
    """Factory returning a ready‑to‑train instance."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybrid"]
