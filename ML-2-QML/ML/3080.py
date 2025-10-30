"""Hybrid classical model combining QCNN and Sampler modules."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)  # output 2 logits for sampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return self.head(x)


class SamplerModule(nn.Module):
    """Softmax head used to interpret QCNN logits as class probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class QCNNSamplerModel(nn.Module):
    """Hybrid model that first extracts features via QCNNModel and then classifies with SamplerModule."""

    def __init__(self) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        self.sampler = SamplerModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qcnn(x)
        return self.sampler(features)


def QCNNSampler() -> QCNNSamplerModel:
    """Factory returning a configured QCNNSamplerModel."""
    return QCNNSamplerModel()


__all__ = ["QCNNSampler", "QCNNSamplerModel", "QCNNModel", "SamplerModule"]
