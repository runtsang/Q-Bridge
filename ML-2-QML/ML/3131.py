"""Hybrid fraud detection model combining CNN feature extraction and photonic‑inspired layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑inspired fully‑connected layer."""
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

class PhotonicLayer(nn.Module):
    """Trainable layer that mirrors the structure of the photonic circuit."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        # Linear part
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

        # Activation and scaling
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

        # Kerr‑like nonlinearity: simple elementwise cubic
        self.kerr = torch.tensor(params.kerr, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        out = out + self.kerr * out.pow(3)
        return out

class FraudDetectionModel(nn.Module):
    """Hybrid fraud detection model: CNN + photonic‑inspired layers + final classifier."""
    def __init__(self,
                 cnn_layers: nn.Module = None,
                 photonic_params: Iterable[FraudLayerParameters] | None = None):
        super().__init__()
        # Feature extractor
        self.features = cnn_layers or nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flatten and project to 2‑dim for photonic layer
        self.project = nn.Linear(16 * 7 * 7, 2)

        # Photonic‑inspired layers
        self.photonic = nn.ModuleList()
        if photonic_params:
            for i, params in enumerate(photonic_params):
                self.photonic.append(PhotonicLayer(params, clip=(i > 0)))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.project(x)
        for layer in self.photonic:
            x = layer(x)
        x = self.classifier(x)
        return x

__all__ = ["FraudLayerParameters", "PhotonicLayer", "FraudDetectionModel"]
