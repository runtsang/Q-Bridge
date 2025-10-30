"""Hybrid fraud detection model with residual connections and autoencoder."""

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
    """Clamp a value to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single layer from parameters, optionally clipping."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model with residual connections and an autoencoder head.

    The model builds a sequential network from a list of ``FraudLayerParameters``.
    A residual connection is added after each layer, and the final output is
    compressed by a one‑dimensional autoencoder before the classification head.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)]
            + [_layer_from_params(layer, clip=True) for layer in layers]
        )
        self.residual = nn.Linear(2, 2)
        self.autoencoder = nn.Sequential(
            nn.Linear(2, 1), nn.Sigmoid(), nn.Linear(1, 2)
        )
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            out = layer(x)
            out = out + self.residual(x)  # residual connection
            x = out
        x = self.autoencoder(x)
        return self.classifier(x)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetectionHybrid:
    """Return an instance of ``FraudDetectionHybrid`` configured with the supplied parameters."""
    return FraudDetectionHybrid(input_params, layers)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
