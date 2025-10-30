"""Upgraded classical fraud detection model with residual connections and regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


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


class ResidualLinearLayer(nn.Module):
    """Linear layer with residual skip connection, batch norm, dropout, and gated scaling."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor,
                 scale: torch.Tensor, shift: torch.Tensor,
                 clip: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(p=0.1)
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.activation(out)
        if self.clip:
            out = out.clamp(-5.0, 5.0)
        out = self.batchnorm(out)
        out = out * self.scale + self.shift
        out = self.dropout(out)
        # Residual addition
        return x + out


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    return ResidualLinearLayer(weight, bias, scale, shift, clip=clip)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model with residual layers and regularization."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
