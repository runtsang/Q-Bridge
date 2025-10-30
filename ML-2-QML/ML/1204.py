"""Enhanced QCNN-inspired network with batch‑norm, dropout, and flexible depth."""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, List, Optional


class QCNNModel(nn.Module):
    """Stack of linear layers with batch‑norm, tanh, and dropout.

    Parameters
    ----------
    input_dim : int, default 8
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int] | None, default None
        Sequence of hidden layer sizes. If ``None`` a sensible default is used.
    dropout : float, default 0.2
        Dropout probability applied after each activation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 12, 8, 4]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                ]
            )
            prev = h
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Return a QCNNModel instance with default configuration."""
    return QCNNModel()


__all__ = ["QCNNModel", "QCNN"]
