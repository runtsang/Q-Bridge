"""Hybrid fraud detection model combining photonic‑inspired layers and QCNN‑style convolution.

This module defines a PyTorch model that mirrors the structure of the
original photonic fraud detection circuit while incorporating the
convolutional and pooling pattern of QCNN.  Each linear block is
initialized with weight and bias clamping to emulate the bounded
photonic parameters, and a scale/shift buffer is applied after the
activation to match the displacement operation in the photonic
analogues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑inspired linear block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip_value(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _photonic_linear(in_features: int, out_features: int, clip: bool = True) -> nn.Module:
    """Create a linear block that mimics a photonic layer."""
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.rand(out_features)
    shift = torch.rand(out_features)

    class PhotonicBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return PhotonicBlock()


class FraudDetectionHybridModel(nn.Module):
    """Hybrid classical model combining photonic‑inspired layers and QCNN structure."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = _photonic_linear(16, 16)
        self.pool1 = _photonic_linear(16, 12)
        self.conv2 = _photonic_linear(12, 8)
        self.pool2 = _photonic_linear(8, 4)
        self.conv3 = _photonic_linear(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def FraudDetectionHybrid() -> FraudDetectionHybridModel:
    """Factory returning a configured :class:`FraudDetectionHybridModel`."""
    return FraudDetectionHybridModel()


__all__ = ["FraudLayerParameters", "FraudDetectionHybridModel", "FraudDetectionHybrid"]
