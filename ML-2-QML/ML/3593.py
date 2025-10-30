from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic‑style layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clip a value to a symmetric bound."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a linear + tanh layer with optional clipping and scale‑shift."""
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

class QCNN__gen223Model(nn.Module):
    """Hybrid QCNN model that mirrors the quantum architecture while adding per‑layer scaling and clipping."""
    def __init__(self) -> None:
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        # Convolution and pooling layers with custom photonic‑style parameters
        self.conv1 = nn.Sequential(_layer_from_params(FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3, phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)), nn.Tanh()))
        self.pool1 = nn.Sequential(_layer_from_params(FraudLayerParameters(
            bs_theta=0.4, bs_phi=0.2, phases=(0.05, -0.05),
            squeeze_r=(0.15, 0.15), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)), nn.Tanh()))
        self.conv2 = nn.Sequential(_layer_from_params(FraudLayerParameters(
            bs_theta=0.3, bs_phi=0.1, phases=(0.02, -0.02),
            squeeze_r=(0.1, 0.1), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)), nn.Tanh()))
        self.pool2 = nn.Sequential(_layer_from_params(FraudLayerParameters(
            bs_theta=0.2, bs_phi=0.05, phases=(0.01, -0.01),
            squeeze_r=(0.05, 0.05), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)), nn.Tanh()))
        self.conv3 = nn.Sequential(_layer_from_params(FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.0, phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)), nn.Tanh()))
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN__gen223() -> QCNN__gen223Model:
    """Factory returning the configured QCNN__gen223Model."""
    return QCNN__gen223Model()

__all__ = ["QCNN__gen223", "QCNN__gen223Model", "FraudLayerParameters"]
