"""Hybrid classical binary classifier with fraud‑inspired parameterized layers.

The model consists of a CNN backbone followed by a small fully‑connected
network built from `FraudLayerParameters`.  Each layer is optionally
clipped, scaled, and shifted, mirroring the photonic fraud detection
architecture.  The final output is passed through a sigmoid to produce
class probabilities.

This module is fully classical and depends only on PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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

def build_fraud_detection_program_ml(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridClassifier(nn.Module):
    """CNN backbone followed by a fraud‑inspired fully‑connected head."""
    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),  # two logits
        )
        # Default fraud‑inspired head with no extra layers
        self.head = build_fraud_detection_program_ml(
            FraudLayerParameters(
                bs_theta=0.1, bs_phi=0.2,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            [],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        # Reduce to a single scalar per sample for the fraud head
        scalar = logits.mean(dim=1, keepdim=True)
        out = self.head(scalar)
        prob = torch.sigmoid(out)
        return torch.cat([prob, 1 - prob], dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program_ml", "HybridClassifier"]
