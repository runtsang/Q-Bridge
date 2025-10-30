"""Combined classical QCNN with parameter clipping and scaling inspired by FraudDetection."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable


class _LinearScaleShift(nn.Module):
    """Linear layer with optional clipping, scaling and shifting."""
    def __init__(self, in_features: int, out_features: int, clip: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.clip:
            self.linear.weight.data.clamp_(-5.0, 5.0)
            self.linear.bias.data.clamp_(-5.0, 5.0)
        x = self.linear(x)
        x = torch.tanh(x)
        x = x * self.scale + self.shift
        return x


class QCNNHybridModel(nn.Module):
    """Classical QCNN with clipping and scaling per layer."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = _LinearScaleShift(8, 16, clip=False)
        self.conv1 = _LinearScaleShift(16, 16)
        self.pool1 = _LinearScaleShift(16, 12)
        self.conv2 = _LinearScaleShift(12, 8)
        self.pool2 = _LinearScaleShift(8, 4)
        self.conv3 = _LinearScaleShift(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning the configured QCNNHybridModel."""
    return QCNNHybridModel()


__all__ = ["QCNNHybrid", "QCNNHybridModel"]
