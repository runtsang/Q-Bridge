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


class FraudDetectionModel(nn.Module):
    """Classical fraud detection model with residual connections and dropout."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [self._layer_from_params(p, clip=False) for p in [input_params] + list(layers)]
        )
        self.final_linear = nn.Linear(2, 1)
        self.dropout = nn.Dropout(p=0.1)

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual  # residual connection
            out = self.dropout(out)
        out = self.final_linear(out)
        return torch.sigmoid(out)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
