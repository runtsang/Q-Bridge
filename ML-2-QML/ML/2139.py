"""
Classical fraud detection model with residual blocks and adaptive training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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


def _linear_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Linear:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    return linear


class ResidualBlock(nn.Module):
    """Residual block that mirrors the photonic layer but adds BN and dropout."""
    def __init__(self, params: FraudLayerParameters, *, clip: bool):
        super().__init__()
        self.linear = _linear_from_params(params, clip=clip)
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = self.bn(out)
        out = self.dropout(out)
        out = out * self.scale + self.shift
        return out + x  # skip connection


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model with residual blocks."""
    modules: list[nn.Module] = [ResidualBlock(input_params, clip=False)]
    modules.extend(ResidualBlock(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
