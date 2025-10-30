"""Enhanced QCNN model with configurable layers and regularisation."""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

class QCNNModel(nn.Module):
    """A configurable fully‑connected network that mimics a QCNN.

    Parameters
    ----------
    input_dim : int
        Size of the input feature vector.
    hidden_dims : Sequence[int] | None
        Sizes of successive hidden layers. The default implements the
        original 8‑>16‑>16‑>12‑>8‑>4‑>4‑>1 architecture.
    activation : nn.Module, optional
        Activation function to apply after each linear block.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    batch_norm : bool, optional
        Whether to insert a ``BatchNorm1d`` after each linear block.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Sequence[int] | None = None,
        activation: nn.Module | None = nn.Tanh(),
        dropout: float | None = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers: list[nn.Module] = []
        in_dim = input_dim
        act = activation if activation is not None else nn.Identity()
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.feature_map = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        return torch.sigmoid(self.head(x))

    def num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def QCNN() -> QCNNModel:
    """Factory that returns a ``QCNNModel`` with default hyper‑parameters."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
