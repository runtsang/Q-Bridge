"""Classical QCNN-inspired residual network with dropout and embedding extraction."""

from __future__ import annotations

import torch
from torch import nn


class QCNNModel(nn.Module):
    """
    A classical QCNN-inspired network.
    Residual skip connections, dropout for regularisation, and a method to retrieve
    the intermediate embedding before the final classification head.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dim // 8, hidden_dim // 8),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim // 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual skip connections after each convolutional block
        y = self.feature_map(x)
        y = self.conv1(y) + y
        y = self.pool1(y)
        y = self.conv2(y) + y
        y = self.pool2(y)
        y = self.conv3(y) + y
        y = self.dropout(y)
        return torch.sigmoid(self.head(y))

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the representation before the final classification head.
        Useful for feature extraction or transfer learning.
        """
        y = self.feature_map(x)
        y = self.conv1(y) + y
        y = self.pool1(y)
        y = self.conv2(y) + y
        y = self.pool2(y)
        y = self.conv3(y) + y
        return y


def QCNN() -> QCNNModel:
    """Factory returning a configured QCNNModel instance."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
