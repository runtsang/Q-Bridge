"""Hybrid QCNN model combining classical fully‑connected layers with quantum‑style parameter clipping."""

from __future__ import annotations

import torch
from torch import nn

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _clipped_linear(in_features: int, out_features: int) -> nn.Linear:
    """Linear layer with weights clipped to [-5,5] and biases to [-5,5]."""
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(torch.clamp(linear.weight, -5.0, 5.0))
        linear.bias.copy_(torch.clamp(linear.bias, -5.0, 5.0))
    return linear

class QCNNGen520(nn.Module):
    """Hybrid QCNN: classical fully‑connected network with parameter clipping inspired by photonic layers."""
    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution‑like layers
        self.conv1 = nn.Sequential(_clipped_linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(_clipped_linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(_clipped_linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(_clipped_linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(_clipped_linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

__all__ = ["QCNNGen520"]
