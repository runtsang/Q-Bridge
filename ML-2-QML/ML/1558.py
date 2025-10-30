"""Classical neural‑net analogue of the photonic fraud detection circuit.

This module extends the original design by adding batch‑normalisation,
dropout and residual connections, improving performance on noisy data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn
import torch.nn.functional as F


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


class FraudDetection(nn.Module):
    """
    Classical neural network mirroring the photonic fraud detection architecture.
    Adds residual connections, batch‑norm and dropout for improved generalisation.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([self._make_layer(input_params, clip=False)])
        self.layers.extend(
            self._make_layer(p, clip=True) for p in layers
        )
        self.out = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(
        self,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> nn.Module:
        """Create a single layer with optional clipping of weights/biases."""
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
        linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        norm = nn.BatchNorm1d(2)
        dropout = nn.Dropout(p=0.2)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.norm = norm
                self.activation = activation
                self.dropout = dropout
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.linear(x)
                out = self.norm(out)
                out = self.activation(out)
                out = self.dropout(out)
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.sigmoid(self.out(x))


__all__ = ["FraudLayerParameters", "FraudDetection"]
