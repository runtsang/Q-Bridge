from __future__ import annotations

import torch
from torch import nn

class QCNNModel(nn.Module):
    """
    Dropout‑regularised, batch‑normalised classical QCNN analogue.
    The architecture mirrors the original quantum convolution steps
    while adding modern best‑practice layers.
    """
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 dropout_rate: float = 0.1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        assert len(hidden_dims) == 6, "expected six hidden dimensions"

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.BatchNorm1d(hidden_dims[3]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dims[3], hidden_dims[4]),
            nn.BatchNorm1d(hidden_dims[4]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[4], hidden_dims[5]),
            nn.BatchNorm1d(hidden_dims[5]),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.head = nn.Linear(hidden_dims[5], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN(**kwargs) -> QCNNModel:
    """
    Factory preserving the original API.
    """
    return QCNNModel(**kwargs)

__all__ = ["QCNN", "QCNNModel"]
