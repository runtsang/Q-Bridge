"""Enhanced QCNN model with residual connections, dropout, and batch normalization.

This module defines a `QCNN` class that inherits from `torch.nn.Module`. It extends the original
convolution-inspired architecture by adding dropout layers, batch normalization, and optional
residual connections. The network is still fully connected but mimics a convolutional pipeline
with pooling and a final classification head.

The class is fully compatible with standard PyTorch training loops and can be dropped into
existing pipelines that expect an `nn.Module`.
"""

import torch
from torch import nn

class QCNN(nn.Module):
    """Convolution-like neural network with residuals, dropout, and batchnorm."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.2, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.BatchNorm1d(12),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(4, 1)

    def _forward_block(self, block, x):
        return block(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        if self.residual:
            x = x + self._forward_block(self.conv1, x)
        else:
            x = self._forward_block(self.conv1, x)
        x = self.pool1(x)
        if self.residual:
            x = x + self._forward_block(self.conv2, x)
        else:
            x = self._forward_block(self.conv2, x)
        x = self.pool2(x)
        if self.residual:
            x = x + self._forward_block(self.conv3, x)
        else:
            x = self._forward_block(self.conv3, x)
        return torch.sigmoid(self.head(x))

    def parameters_summary(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layers(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = not freeze

__all__ = ["QCNN"]
