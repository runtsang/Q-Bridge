"""Classical QCNN with residual connections, dropout, and batch‑norm."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class _ResBlock(nn.Module):
    """A residual block that adds its input to the output of a linear layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        # Pad if dimensions differ
        if x.shape[-1]!= out.shape[-1]:
            pad = torch.zeros_like(x)
            pad[..., :out.shape[-1]] = x[..., :out.shape[-1]]
            x = pad
        return x + out


class QCNNModel(nn.Module):
    """
    Stack of fully‑connected layers that emulate a quantum convolutional neural network.
    Residual connections allow gradients to flow more easily through the network, while
    dropout and batch‑norm regularise the model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16), nn.BatchNorm1d(16), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.conv1 = _ResBlock(16, 16)
        self.pool1 = _ResBlock(16, 12)
        self.conv2 = _ResBlock(12, 8)
        self.pool2 = _ResBlock(8, 4)
        self.conv3 = _ResBlock(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


__all__ = ["QCNN", "QCNNModel"]
