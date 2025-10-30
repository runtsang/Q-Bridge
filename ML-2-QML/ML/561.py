"""Extended classical QCNN with residual blocks, batch‑norm and dropout.

The architecture mirrors the original QCNN but introduces
residual connections, batch‑normalisation and dropout to improve
generalisation.  The public API remains unchanged – the factory
function ``QCNNExtended`` returns a ready‑to‑train instance.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """One‑layer residual unit used in the convolutional stages."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class QCNNExtendedModel(nn.Module):
    """Classical QCNN with residual layers and regularisation."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )
        self.conv1 = ResidualBlock(hidden_dims[0], hidden_dims[1], dropout)
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
        )
        self.conv2 = ResidualBlock(hidden_dims[2], hidden_dims[3], dropout)
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.BatchNorm1d(hidden_dims[4]),
            nn.ReLU(),
        )
        self.conv3 = ResidualBlock(hidden_dims[4], hidden_dims[5], dropout)
        self.head = nn.Linear(hidden_dims[5], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNNExtended() -> QCNNExtendedModel:
    """Return a freshly‑initialised QCNNExtendedModel."""
    return QCNNExtendedModel()


__all__ = ["QCNNExtended", "QCNNExtendedModel"]
