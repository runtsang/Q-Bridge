"""Hybrid fraud detection model combining photonic-inspired layers and QCNN-style architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, reused from the original seed."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _photonic_layer_from_params(params: FraudLayerParameters, output_dim: int = 8) -> nn.Module:
    """Create a linear layer that mimics a photonic layer using the supplied parameters."""
    # Build a weight matrix of shape (output_dim, 2) by repeating the 2x2 core
    core = torch.tensor([[params.bs_theta, params.bs_phi],
                         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    weight = core.repeat(output_dim // 2, 1)
    bias = torch.zeros(output_dim, dtype=torch.float32)
    # Clip weights and bias for numerical stability
    weight = weight.clamp(-5.0, 5.0)
    bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, output_dim)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32).repeat(output_dim // 2, 1).flatten()
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32).repeat(output_dim // 2, 1).flatten()

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class QCNNModel(nn.Module):
    """Classical network that emulates the QCNN structure."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridFraudDetectionModel(nn.Module):
    """Hybrid model that first extracts features via a photonic-inspired layer
    and then processes them with a QCNN-style network."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        # Photonic feature extractor
        self.photonic = _photonic_layer_from_params(input_params, output_dim=8)
        # QCNN core
        self.qcnn = QCNNModel()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.photonic(inputs)
        return self.qcnn(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> HybridFraudDetectionModel:
    """Construct a hybrid fraud detection model from photonic and QCNN parameters."""
    return HybridFraudDetectionModel(input_params, layers)


__all__ = [
    "FraudLayerParameters",
    "HybridFraudDetectionModel",
    "build_fraud_detection_program",
]
