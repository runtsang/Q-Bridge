"""Hybrid fraud detection model combining classical CNN and quantum‑inspired layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

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

class FraudDetectionHybrid(nn.Module):
    """Hybrid CNN + quantum‑inspired surrogate model for fraud detection."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        image_size: int = 28,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        # Classical feature extractor – inspired by Quantum‑NAT
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=conv_kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=conv_kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce to a 2‑dimensional vector for the surrogate quantum block
        flat_size = 16 * (image_size // 4) ** 2
        self.reduce = nn.Linear(flat_size, 2)
        # Quantum‑inspired surrogate layers
        self.quantum_surrogate = nn.Sequential(
            *[self._layer_from_params(p, clip=True) for p in layers]
        )
        self.classifier = nn.Linear(2, 1)
        self.norm = nn.BatchNorm1d(1)

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
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                outputs = self.activation(self.linear(inputs))
                return outputs * self.scale + self.shift

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        flat = feat.view(x.size(0), -1)
        reduced = self.reduce(flat)
        q_out = self.quantum_surrogate(reduced)
        out = self.classifier(q_out)
        return self.norm(out)

def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    image_size: int = 28,
) -> FraudDetectionHybrid:
    """Convenience factory matching the original anchor signature."""
    return FraudDetectionHybrid(input_params, layers, image_size=image_size)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybrid",
    "build_fraud_detection_model",
]
