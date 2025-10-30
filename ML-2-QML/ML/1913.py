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
    dropout_prob: float = 0.0  # new feature: optional dropout

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    """Create a single Fraud layer with optional clipping and dropout."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    dropout = nn.Dropout(params.dropout_prob) if params.dropout_prob > 0 else None

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
            if dropout is not None:
                self.dropout = dropout

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            out = torch.clamp(out, -5.0, 5.0) if clip else out
            out = out * self.scale + self.shift
            if dropout is not None:
                out = self.dropout(out)
            return out

    return Layer()

class ResidualFraudLayer(nn.Module):
    """A residual wrapper around a Fraud layer."""
    def __init__(self, params: FraudLayerParameters, *, clip: bool) -> None:
        super().__init__()
        self.fraud = _layer_from_params(params, clip=clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fraud(x)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Assemble a sequential PyTorch model.

    The first layer is a plain Fraud layer; subsequent layers are residual.
    The final layer is a sigmoid classifier.
    """
    modules: Sequence[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules = list(modules)

    for layer_params in layers:
        modules.append(ResidualFraudLayer(layer_params, clip=True))

    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())

    return nn.Sequential(*modules)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
