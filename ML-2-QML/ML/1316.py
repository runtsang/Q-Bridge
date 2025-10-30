"""Enhanced classical QCNN model with residual connections, batch‑norm, and dropout.

This module defines :class:`QCNNModel`, a PyTorch neural network that
mirrors the structure of the seed QCNN but adds modern regularisation
techniques.  The model is fully compatible with the training utilities
of the original seed and can be dropped into a scikit‑learn pipeline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from typing import Iterable, List, Tuple

class QCNNModel(nn.Module):
    """
    A fully‑connected neural network that emulates the quantum‑convolution
    layers of the original QCNN while adding residual connections,
    batch‑normalisation and dropout for improved generalisation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: Iterable[int] | None = None,
        dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers: List[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.body = nn.Sequential(*layers)
        self.residual = residual
        if residual:
            # identity mapping for residual skip
            self.res_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.body(x)
        if self.residual:
            # broadcast residual to match output shape
            out = out + self.res_layer(residual)
        return torch.sigmoid(out)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """
        One training step.  Returns the scalar loss value.
        """
        self.train()
        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

def QCNN() -> QCNNModel:
    """
    Factory that returns a ready‑to‑train :class:`QCNNModel` instance.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
