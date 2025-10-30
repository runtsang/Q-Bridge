"""Enhanced classical fraud detection model with dropout, residual connections, and sigmoid output."""

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
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool, dropout_prob: float = 0.1
) -> nn.Module:
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
    dropout = nn.Dropout(p=dropout_prob)
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = dropout
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.linear(inputs)
            out = self.activation(out)
            out = self.dropout(out)
            out = out * self.scale + self.shift
            # Residual connection
            return out + inputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *, dropout_prob: float = 0.1,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure with dropout and sigmoid output."""
    modules = [_layer_from_params(input_params, clip=False, dropout_prob=dropout_prob)]
    modules.extend(
        _layer_from_params(layer, clip=True, dropout_prob=dropout_prob) for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


class FraudDetection:
    """Shared interface for classical fraud detection model."""
    @staticmethod
    def build_model(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_prob: float = 0.1,
    ) -> nn.Module:
        return build_fraud_detection_program(input_params, layers, dropout_prob=dropout_prob)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetection"]
