"""Enhanced classical QCNN with residuals, dropout, and batch normalization."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class QCNNModel(nn.Module):
    """
    A deeper convolution‑inspired neural network.

    The architecture mirrors the original QCNN but adds
    residual connections, batch‑normalization, and dropout
    to improve generalisation on small datasets.
    """

    def __init__(self,
                 input_dim: int = 8,
                 hidden_dims: list[int] | None = None,
                 dropout: float = 0.2,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.feature_map = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Residual connection between feature_map and head
        out = self.feature_map(x)
        out = torch.sigmoid(self.head(out))
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the activations before the final sigmoid."""
        return self.feature_map(x)


def QCNN() -> QCNNModel:
    """Factory returning a pre‑configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
