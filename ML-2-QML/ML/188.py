"""Extended classical fraud detection model with residual connections and sigmoid output."""

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


class ResidualBlock(nn.Module):
    """Adds the input to the output of a submodule to create a residual connection."""
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) + x


class FraudDetectionModel(nn.Module):
    """Classical fraud detection model with residual connections and a sigmoid output."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        layers = list(layers)

        def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

                def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                    outputs = self.activation(self.linear(inputs))
                    outputs = outputs * self.scale + self.shift
                    return outputs

            return Layer()

        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(ResidualBlock(_layer_from_params(l, clip=True)) for l in layers)
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return self.sigmoid(logits)

    @staticmethod
    def build_fraud_detection_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> "FraudDetectionModel":
        """Convenience constructor mirroring the original seed API."""
        return FraudDetectionModel(input_params, layers)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
