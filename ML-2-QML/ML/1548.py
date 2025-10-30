"""Classical fraud‑detection model with dropout, batch‑norm and clipping.

The implementation mirrors the photonic architecture but adds
regularisation and a convenient `FraudDetectionModel` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    # optional regularisers
    dropout: float = 0.0
    batchnorm: bool = False


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    # Construct a linear layer with the given weights/biases.
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

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.bn = nn.BatchNorm1d(2) if params.batchnorm else nn.Identity()
            self.activation = nn.Tanh()
            self.dropout = nn.Dropout(params.dropout) if params.dropout > 0.0 else nn.Identity()
            self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
            self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(x)
            x = self.bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionModel(nn.Module):
    """Hybrid‑style fraud‑detection network with optional regularisers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(_layer_from_params(input_params, clip=False))
        for layer in layers:
            self.layers.append(_layer_from_params(layer, clip=clip))
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    clip: bool = True,
) -> nn.Sequential:
    """Compatibility wrapper that returns a sequential model."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=clip) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
    "build_fraud_detection_program",
]
