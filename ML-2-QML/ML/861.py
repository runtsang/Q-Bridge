"""
Classical fraud detection model with residual, dropout, and batch‑normalization layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn, Tensor


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel(nn.Module):
    """
    Classical neural network mirroring a photonic fraud detection circuit.
    Adds residual connections, dropout, and optional batch‑normalization for robustness.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: List[FraudLayerParameters],
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.input_layer = self._layer_from_params(input_params, clip=False)
        self.hidden_layers = nn.ModuleList(
            [self._layer_from_params(l, clip=True) for l in layers]
        )
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(2) if use_batchnorm else nn.Identity()
        self.output_layer = nn.Linear(2, 1)

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

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
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: Tensor) -> Tensor:
                y = self.activation(self.linear(x))
                y = y * self.scale + self.shift
                return y

        return Layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.batchnorm(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
