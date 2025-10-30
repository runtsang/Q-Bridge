"""Enhanced classical fraud detection model with residual connections and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model, extended with regularization options."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout_rate: float = 0.0
    batch_norm: bool = False


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
            self.bn = nn.BatchNorm1d(2) if params.batch_norm else nn.Identity()
            self.dropout = nn.Dropout(params.dropout_rate) if params.dropout_rate > 0.0 else nn.Identity()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.linear(inputs)
            outputs = self.activation(outputs)
            outputs = self.bn(outputs)
            outputs = self.dropout(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


class FraudDetectionHybrid(nn.Module):
    """Hybrid classical fraud detection model with optional residual connections."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.main = build_fraud_detection_program(input_params, layers)
        if residual:
            self.residual_conv = nn.Linear(2, 2)
            self.residual_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        if self.residual:
            res = self.residual_activation(self.residual_conv(x))
            out = out + res
        return out


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
