"""Hybrid classical model combining CNN feature extractor and fraud‑detection style parameterized layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Sequence

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

class HybridNATModel(nn.Module):
    """Hybrid classical model combining convolutional feature extraction with fraud‑detection style parameterized layers."""

    def __init__(
        self,
        fraud_params: Sequence[FraudLayerParameters],
        num_classes: int = 4,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fraud‑style parameterized layers
        fraud_modules: list[nn.Module] = [_layer_from_params(fraud_params[0], clip=False)]
        fraud_modules.extend(_layer_from_params(p, clip=True) for p in fraud_params[1:])
        self.fraud_head = nn.Sequential(*fraud_modules, nn.Linear(2, 1))
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        # Use first channel as fraud input; reshape to (bsz, 2)
        fraud_input = x[:, :1, :, :].view(bsz, -1)[:, :2]
        fraud_out = self.fraud_head(fraud_input)
        combined = torch.cat([flattened, fraud_out], dim=1)
        out = self.fc(combined)
        return self.norm(out)

__all__ = ["HybridNATModel"]
