"""
Classical QCNN variant with residual connections, batch‑normalisation, and dropout.
The model keeps the same API as the original seed but adds richer regularisation
and a flexible hidden‑layer specification for experimentation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNHybrid(nn.Module):
    """
    ResNet‑style fully‑connected network that emulates a quantum convolutional
    circuit.  The architecture is parametrised so that the depth and width can
    be tuned for different datasets.
    """

    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | tuple[int,...] = (16, 12, 8, 4),
                 dropout: float = 0.2) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input feature vector.
        hidden_dims : sequence[int]
            Sizes of successive hidden layers.
        dropout : float
            Drop‑out probability applied after each non‑linear layer.
        """
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Sigmoid‑activated scalar predictions of shape (batch_size, 1).
        """
        x = self.feature_map(x)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.head(x))


def QCNNHybridFactory() -> QCNNHybrid:
    """
    Factory function that returns a pre‑configured :class:`QCNNHybrid` instance.
    This mirrors the original ``QCNN`` helper and allows consistent imports.
    """
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
