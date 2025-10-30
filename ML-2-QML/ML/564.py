"""Hybrid fraud detection model with residual connections and regularisation.

This module extends the original classical analogue by adding batch‑normalisation,
drop‑out, Swish activation and a residual connection between successive layers.
The model retains the same `FraudLayerParameters` interface so that it can be
instantiated with the same configuration dictionary used for the photonic
counterpart.

The resulting `FraudDetectionHybrid` is a pure PyTorch `nn.Module` that can be
used in any standard training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

@dataclass
class FraudLayerParameters:
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

def _create_layer(params: FraudLayerParameters, clip: bool) -> nn.Module:
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

    activation = nn.SiLU()  # Swish
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    dropout = nn.Dropout(p=0.1)
    batchnorm = nn.BatchNorm1d(2)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = dropout
            self.batchnorm = batchnorm
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.linear(x)
            out = self.activation(out)
            out = self.batchnorm(out)
            out = self.dropout(out)
            out = out * self.scale + self.shift
            return out

    return Layer()

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model with residual connections and dropout."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        super().__init__()
        self.layers = nn.ModuleList(
            [_create_layer(input_params, clip=False)] +
            [_create_layer(l, clip=True) for l in layers]
        )
        self.residual = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out) + self.residual(out)  # residual shortcut
        out = self.classifier(out)
        return out

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
