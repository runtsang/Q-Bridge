"""Enhanced classical fraud detection model with residual and dropout layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


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


class _FraudLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool = True, dropout: float = 0.2):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.activation(y)
        y = self.bn(y)
        y = self.dropout(y)
        y = y * self.scale + self.shift
        return y


class FraudDetectionModel(nn.Module):
    """Classical fraud detection model with residual connections and dropout."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_layer = _FraudLayer(input_params, clip=False, dropout=0.0)
        self.hidden_layers = nn.ModuleList(
            [_FraudLayer(p, clip=True, dropout=dropout) for p in layers]
        )
        self.output = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out) + out
        out = self.output(out)
        return torch.sigmoid(out)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
