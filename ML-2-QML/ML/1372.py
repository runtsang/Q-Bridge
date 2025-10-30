"""Hybrid QCNN-inspired model with residual connections and dropout.

Extends the original QCNNModel by adding batch normalization after each
linear block, a dropout layer, and a residual skip connection between
convolutional blocks. This improves expressivity and regularisation
while keeping the architecture lightweight.
"""

import torch
from torch import nn


class QCNNHybrid(nn.Module):
    """QCNN-inspired feed‑forward network with residuals and dropout."""

    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )
        self.conv_blocks = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Tanh(),
            )
            self.conv_blocks.append(block)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.conv_blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = self.dropout(x)
        return torch.sigmoid(self.head(x))


def create_QCNNHybrid() -> QCNNHybrid:
    """Factory that returns a ready‑to‑train QCNNHybrid model."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "create_QCNNHybrid"]
