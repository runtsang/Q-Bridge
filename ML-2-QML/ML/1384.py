"""Enhanced classical fraud detection model with dropout and batch‑norm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

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


class ScaleShift(nn.Module):
    """Applies element‑wise scaling and shifting."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: Optional[float] = None,
) -> nn.Module:
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

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    layers = [linear, activation]
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.BatchNorm1d(2))
    layers.append(ScaleShift(scale, shift))

    return nn.Sequential(*layers)


class FraudDetectionHybrid(nn.Module):
    """Extended fraud detection model with optional dropout and batch‑norm."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: Optional[float] = None,
        final_bias: bool = True,
    ) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False, dropout=dropout)]
        modules.extend(
            _layer_from_params(layer, clip=True, dropout=dropout) for layer in layers
        )
        modules.append(nn.Linear(2, 1, bias=final_bias))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
