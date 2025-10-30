"""Enhanced classical fraud detection model with dropout and batch normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn

@dataclass
class FraudLayerParameters:
    """Parameters for a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    dropout_rate: float = 0.0
    batch_norm: bool = False

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to ``[-bound, bound]``."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
            if params.batch_norm:
                self.bn = nn.BatchNorm1d(2)
            else:
                self.bn = nn.Identity()
            self.dropout = nn.Dropout(params.dropout_rate)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.bn(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with dropout and batch‑norm."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionEnhanced:
    """Wrapper that builds the classical fraud‑detection network and exposes a predict API."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.model = build_fraud_detection_program(input_params, layers)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fraud probability after sigmoid."""
        logits = self.model(x)
        return torch.sigmoid(logits)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionEnhanced"]
