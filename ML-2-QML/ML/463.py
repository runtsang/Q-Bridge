"""Upgraded classical fraud detection model with dropout and residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected layer."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0  # new field for regularisation


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the interval [‑bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single layer with optional clipping and dropout."""
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
    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
    dropout = nn.Dropout(params.dropout) if params.dropout > 0 else nn.Identity()

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = dropout
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            # Residual connection
            return x + inputs

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Build a sequential PyTorch model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (no clipping).
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers (clipped to prevent exploding weights).

    Returns
    -------
    nn.Sequential
        Fully‑connected model ending with a single output neuron.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_model"]
