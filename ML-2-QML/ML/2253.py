"""
HybridQCNet – classical analogue of the hybrid quantum binary classifier.
It combines convolutional feature extraction with a photonic‑style
fully‑connected layer (FraudLayer) and a final sigmoid head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np

__all__ = ["FraudLayerParameters", "FraudLayer", "HybridQCNet"]


@dataclass
class FraudLayerParameters:
    """Parameters for the photonic‑style fully‑connected layer."""
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


class FraudLayer(nn.Module):
    """
    Classical layer that mirrors the structure of a photonic circuit.
    Uses a 2×2 linear transform followed by Tanh, then applies a
    learnable scale and shift (mimicking displacement and squeezing).
    """

    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
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

        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.activation(self.linear(inputs))
        return x * self.scale + self.shift


class HybridQCNet(nn.Module):
    """
    Convolutional network followed by a fraud‑layer and a sigmoid head.
    Mirrors the architecture of the quantum version but uses a classical
    final layer, enabling fast training and serving as a baseline.
    """

    def __init__(self,
                 fraud_params: FraudLayerParameters,
                 fraud_clip: bool = True) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected blocks
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2 outputs for fraud layer

        # Fraud‑style layer
        self.fraud = FraudLayer(fraud_params, clip=fraud_clip)

        # Final linear head
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.fraud(x)
        logits = self.out(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)
