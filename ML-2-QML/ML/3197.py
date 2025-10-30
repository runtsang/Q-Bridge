"""Hybrid fraud detection model combining photonic-inspired layers and CNN feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameter container for a photonic‑inspired classical layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _layer_from_params(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
    """Create a single layer that mimics a photonic circuit using linear and activation."""
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """Hybrid model: CNN feature extractor followed by photonic-inspired layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Sequence[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # CNN feature extractor identical to the Quantum‑NAT CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Project to 2 channels for the photonic layers
        self.project = nn.Linear(16 * 7 * 7, 2)
        # Photonic‑inspired layers
        self.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)]
            + [_layer_from_params(p, clip=True) for p in hidden_params]
        )
        # Final classifier
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.project(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
