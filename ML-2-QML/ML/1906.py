"""Improved classical fraud detection model with dropout and batch‑norm.

The model is a drop‑in replacement for the original procedural builder
and supports a flexible number of layers, optional dropout, and batch
normalisation.  It is deliberately kept lightweight so that it can be
used as a component in larger PyTorch training pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    """Helper to clip values to a given symmetric bound."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer from the dataclass."""
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
    activation = nn.Sigmoid()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionModel(nn.Module):
    """Hybrid classical fraud‑detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer – these are *not* clipped.
    layer_params : Iterable[FraudLayerParameters]
        Parameters for subsequent hidden layers – these are clipped.
    dropout : float, default 0.1
        Drop‑out probability after each hidden layer.
    batch_norm : bool, default True
        Whether to insert a BatchNorm1d layer after each hidden layer.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        dropout: float = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [_layer_from_params(input_params, clip=False)]
        layers += [_layer_from_params(p, clip=True) for p in layer_params]
        seq = []
        for layer in layers:
            seq.append(layer)
            if batch_norm:
                seq.append(nn.BatchNorm1d(2))
            seq.append(nn.Dropout(dropout))
        seq.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
