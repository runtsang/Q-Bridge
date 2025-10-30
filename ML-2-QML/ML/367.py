"""
QCNNModel – a classical neural network that emulates the depth‑wise convolution
and pooling operations of a quantum convolutional neural network.  The architecture
has been extended with residual connections, batch‑normalization, and dropout
to improve expressivity and regularisation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["QCNN"]


class ResidualBlock(nn.Module):
    """
    A two‑layer fully‑connected residual block with batch‑normalisation.

    Parameters
    ----------
    in_features : int
        Size of the input feature vector.
    out_features : int
        Size of the output feature vector.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        # Shortcut connection – linear if dimensions differ
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(
            in_features, out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class QCNN(nn.Module):
    """
    Classical QCNN with residual blocks, batch‑normalisation and dropout.

    Parameters
    ----------
    input_dim : int, default=8
        Dimensionality of the input vector.
    hidden_dims : list[int], default=[16, 12, 8, 4, 4]
        Width of each successive residual block.
    dropout : float, default=0.2
        Dropout probability applied after the residual stack.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [16, 12, 8, 4, 4]
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.blocks.append(ResidualBlock(in_dim, out_dim))

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Sigmoid‑activated probability of shape ``(batch_size, 1)``.
        """
        x = self.feature_map(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.head(x)
        return torch.sigmoid(x)


def QCNN() -> QCNN:
    """
    Factory that returns a freshly initialised :class:`QCNN` instance.
    """
    return QCNN()
