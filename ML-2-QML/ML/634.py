"""Classical fraud detection model with residual connections and dropout.

The module extends the original seed by adding batch normalization,
dropout, and a simple residual connection.  The resulting
FraudDetectionModel can be trained end‑to‑end with the same
parameter representation used by the quantum counterpart.
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


def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
    dropout_p: float = 0.0,
) -> nn.Module:
    """Create a single linear layer with optional clipping, batch‑norm and dropout."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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
    batchnorm = nn.BatchNorm1d(2)
    dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.batchnorm = batchnorm
            self.dropout = dropout
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            out = self.batchnorm(out)
            out = self.dropout(out)
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionModel(nn.Module):
    """Hybrid classical model with residual connections and dropout."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_layer = _layer_from_params(input_params, clip=False, dropout_p=0.0)
        self.hidden_layers = nn.ModuleList(
            [
                _layer_from_params(l, clip=True, dropout_p=dropout_p)
                for l in layers
            ]
        )
        self.residual = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = out + self.residual(out)  # simple residual
        out = torch.sigmoid(self.classifier(out))
        return out

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
