"""Enhanced classical fraud detection model with residual connections and dropout."""

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
    dropout: float = 0.0  # new field for regularisation


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
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
            self.dropout = nn.Dropout(params.dropout) if params.dropout > 0.0 else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            outputs = self.dropout(outputs)
            return outputs

    return Layer()


class FraudDetectionModel(nn.Module):
    """Classical fraud detection model with optional residual connections and dropout."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        sequential = [_layer_from_params(input_params, clip=False)]
        sequential.extend(_layer_from_params(layer, clip=True) for layer in layers)
        sequential.append(nn.Linear(2, 1))
        self.core = nn.Sequential(*sequential)
        if residual:
            self.res_block = nn.ModuleList(
                [_layer_from_params(layer, clip=True) for layer in layers]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        if self.residual:
            residual = x
            for block in self.res_block:
                residual = block(residual)
            out = out + residual
        return torch.sigmoid(out)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of fraud."""
        return self.forward(x)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
