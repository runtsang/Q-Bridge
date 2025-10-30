"""
High‑performance classical QCNN with regularisation and feature extraction.

The architecture mirrors the original seed but adds
* BatchNorm1d after every linear block
* Dropout (p=0.2) preceding the final head
* A ``feature_extractor`` method that returns the hidden representation
  for downstream tasks (e.g. transfer learning, visualisation).

The design is intentionally lightweight so that the module can be
integrated into any PyTorch training pipeline without modification.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class QCNNModel(nn.Module):
    """
    Fully‑connected network that emulates a quantum convolutional neural
    network.  Each block consists of Linear → BatchNorm → Tanh.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: Tuple[int,...] = (16, 16, 12, 8, 4, 4)) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.Tanh()])
            prev_dim = dim
        self.body = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        x = self.dropout(x)
        return torch.sigmoid(self.head(x))

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the hidden representation before the output layer.
        Useful for transfer learning or visualisation.
        """
        return self.body(x)


def QCNN() -> QCNNModel:
    """
    Factory returning a ready‑to‑train :class:`QCNNModel` instance.
    """
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
