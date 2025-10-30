"""Enhanced classical fraud detection model with dropout and batch normalization.

The original seed provided a simple two-layer fully connected network. This
upgrade adds:
  * Dropout after every activation for regularisation.
  * BatchNorm to stabilise training.
  * A convenience ``from_parameters`` constructor that mimics the quantum
    parameter style.
"""

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
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.bn = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(p=0.1)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.bn(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """Full classical fraud‑detection network built from the photonic parameters."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)

    @classmethod
    def from_parameters(cls, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> "FraudDetectionHybrid":
        """Convenience constructor that mimics the quantum API."""
        return cls(input_params, layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
