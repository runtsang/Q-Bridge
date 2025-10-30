"""Importable classical fraud detection model with extended capabilities.

The module defines a dataclass `FraudLayerParameters` (identical to the seed)
and a class `FraudDetectionEnhanced` that builds a multi‑layer neural
network with batch‑norm, dropout and a final linear classifier.  The
class exposes a `forward` method and a small training helper.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetectionEnhanced(nn.Module):
    """Classical fraud detection model with optional dropout and batch‑norm."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        for layer in layers:
            modules.append(_layer_from_params(layer, clip=True))
        modules.append(nn.BatchNorm1d(2))
        modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        logits = torch.sigmoid(self.forward(x))
        return (logits > threshold).float()

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
