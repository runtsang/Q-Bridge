"""Classical fraud‑detection model with residual learning and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

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


class FraudDetectionHybrid(nn.Module):
    """
    A PyTorch model that mirrors the layered structure of the photonic circuit
    while adding residual connections and dropout for improved generalisation.
    """

    def __init__(
        self,
        layer_params: List[FraudLayerParameters],
        dropout: float = 0.1,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_self._layer_from_params(p, clip=clip) for p in layer_params]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(2, 1)

    @staticmethod
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

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return Layer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and dropout.
        """
        for layer in self.layers:
            residual = layer(inputs)
            inputs = inputs + residual
            inputs = self.dropout(inputs)
        return self.output_layer(inputs)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
