"""QCNNEnhanced: Classical neural network mimicking a quantum convolutional neural network with residuals and modular encoders."""
from __future__ import annotations

import torch
from torch import nn
from typing import Optional

class LinearEncoder(nn.Module):
    """Simple linear encoder with optional dropout."""
    def __init__(self, in_features: int = 8, out_features: int = 8, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.drop = nn.Dropout(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.linear(x))

class QCNNEnhanced(nn.Module):
    """
    Classical neural network that mirrors a quantum convolutional neural network.
    Features:
      - Modular encoder (default LinearEncoder)
      - Residual connections for better gradient flow
      - Dropout for regularisation
      - Optional weight scaling to stabilise training
    """
    def __init__(self,
                 encoder: Optional[nn.Module] = None,
                 dropout: float = 0.0,
                 weight_scale: float = 1.0) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else LinearEncoder()
        self.conv1 = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.weight_scale = weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode input
        x = self.encoder(x)
        # Residual connection
        residual = x
        x = self.conv1(x)
        x += residual
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.pool2(x)
        out = torch.sigmoid(self.head(x)) * self.weight_scale
        return out

def QCNN() -> QCNNEnhanced:
    """Factory returning a QCNNEnhanced model with default settings."""
    return QCNNEnhanced()

__all__ = ["QCNN", "QCNNEnhanced"]
