"""Enhanced classical fraud detection model with dropout and batch normalization.

This module defines a PyTorch neural network that mirrors the structure of the
original seed but adds regularization layers and a flexible initialization
scheme.  The model accepts a list of :class:`FraudLayerParameters` and
constructs a sequential network that can be trained end‑to‑end with any
PyTorch optimiser.
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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool, dropout: float) -> nn.Module:
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
            self.dropout = nn.Dropout(dropout)
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

class FraudDetectionModel(nn.Module):
    """Hybrid classical fraud detection network with regularisation.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) layer.
    layers : Iterable[FraudLayerParameters]
        Subsequent layers that are clipped and regularised.
    dropout : float, default 0.1
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [_layer_from_params(input_params, clip=False, dropout=0.0)]
        modules.extend(
            _layer_from_params(layer, clip=True, dropout=dropout) for layer in layers
        )
        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
