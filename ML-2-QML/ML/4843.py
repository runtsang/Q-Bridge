"""Hybrid classical network combining regression and convolutional structure."""
from __future__ import annotations

import torch
from torch import nn


class HybridEstimator(nn.Module):
    """A hybrid feed‑forward network with convolution‑like layers and a regression head."""
    def __init__(self, input_dim: int = 8, n_classes: int = 2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.reg_head = nn.Linear(4, 1)
        self.class_head = nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        reg = torch.sigmoid(self.reg_head(x))
        cls = self.class_head(x)
        return reg, cls


def EstimatorQNN() -> HybridEstimator:
    """Return the configured hybrid estimator."""
    return HybridEstimator()


__all__ = ["EstimatorQNN"]
