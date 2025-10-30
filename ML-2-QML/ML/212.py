"""Classical fraud detection model with configurable depth and regularisation."""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout: float,
    batch_norm: bool,
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
    linear = nn.Linear(2, 2, bias=True)
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
            if batch_norm:
                self.bn = nn.BatchNorm1d(2)
            if dropout > 0.0:
                self.drop = nn.Dropout(dropout)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = x * self.scale + self.shift
            if hasattr(self, "bn"):
                x = self.bn(x)
            if hasattr(self, "drop"):
                x = self.drop(x)
            return x

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *,
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> nn.Sequential:
    """Build a sequential PyTorch model mirroring the layered structure with optional regularisation."""
    modules = [
        _layer_from_params(input_params, clip=False, dropout=dropout, batch_norm=batch_norm)
    ]
    modules.extend(
        _layer_from_params(layer, clip=True, dropout=dropout, batch_norm=batch_norm)
        for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionModel:
    """Encapsulates the classical fraud detection neural network."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        self.model = build_fraud_detection_program(
            input_params, layers, dropout=dropout, batch_norm=batch_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
]
