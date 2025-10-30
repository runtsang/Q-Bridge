"""Hybrid classical QCNN model combining convolution, pooling, batch‑norm and dropout.

This implementation fuses the convolution‑pooling architecture from the original QCNN seed with
regularisation techniques inspired by Quantum‑NAT and FraudDetection.  The model is fully
PyTorch‑compatible and ready for integration into standard training pipelines.
"""

from __future__ import annotations

import torch
from torch import nn


class QCNNModel(nn.Module):
    """Classical QCNN with batch‑normalisation and dropout.

    The network consists of a feature map followed by three
    convolution–pooling stages and a fully‑connected head.
    Batch‑norm layers are inserted after each linear transform to
    stabilise training, while a dropout layer mitigates over‑fitting.
    """

    def __init__(self) -> None:
        super().__init__()

        self.feature_map = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Tanh(),
        )

        self.conv1 = nn.Sequential(
            nn.Linear(16, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12, bias=False),
            nn.BatchNorm1d(12),
            nn.Tanh(),
        )

        self.conv2 = nn.Sequential(
            nn.Linear(12, 8, bias=False),
            nn.BatchNorm1d(8),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.Tanh(),
        )

        self.conv3 = nn.Sequential(
            nn.Linear(4, 4, bias=False),
            nn.BatchNorm1d(4),
            nn.Tanh(),
        )

        self.head = nn.Linear(4, 1, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.out_norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.head(x)
        return torch.sigmoid(self.out_norm(x))


def QCNN() -> QCNNModel:
    """Factory returning a fully configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
